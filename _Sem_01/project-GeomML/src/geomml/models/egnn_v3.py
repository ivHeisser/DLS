import torch
import torch.nn as nn
from egnn_pytorch import EGNN_Network
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from geomml.utils.base_model import BaseModel
from geomml.registry import MODELS
from torch_geometric.nn import global_mean_pool
from torch_cluster import radius_graph



class RBFExpansion(nn.Module):
# ============================================================
# RBF Expansion
# ============================================================
    def __init__(self,num_rbf=32,cutoff=5.0,gamma=None):
        super().__init__()
        self.cutoff=cutoff
        self.register_buffer("centers",torch.linspace(0,cutoff,num_rbf))
        self.gamma=gamma if gamma is not None else num_rbf/cutoff

    def forward(self,d):
        d=d.clamp(max=self.cutoff)
        return torch.exp(-self.gamma*(d-self.centers)**2)



class AtomEncoder(nn.Module):
# ============================================================
# Atom Encoder
# ============================================================
    def __init__(self,hidden_dim):
        super().__init__()
        self.atom_emb=nn.Embedding(100,hidden_dim)
        self.charge_emb=nn.Embedding(21,hidden_dim)
        self.arom_emb=nn.Embedding(2,hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim)

    def forward(self,z,charge=None,aromatic=None):
        h=self.atom_emb(z.long())

        if charge is not None:
            charge=(charge.long()+10).clamp(0,20)
            h=h+self.charge_emb(charge)

        if aromatic is not None:
            aromatic=aromatic.long().clamp(0,1)
            h=h+self.arom_emb(aromatic)

        return self.norm(h)



class EdgeEncoder(nn.Module):
# ============================================================
# Bond Encoder
# ============================================================
    def __init__(self,hidden_dim,num_rbf=32,bond_dim=8):
        super().__init__()
        self.rbf=RBFExpansion(num_rbf)
        self.rbf_proj=nn.Sequential(nn.Linear(num_rbf,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim))
        self.bond_emb=nn.Embedding(bond_dim,hidden_dim)

    def forward(self,dist,bond_type=None):
        e=self.rbf_proj(self.rbf(dist))
        if bond_type is not None:
            e=e+self.bond_emb(bond_type.long())
        return e


class AngleAwareBlock(nn.Module):
# ============================================================
# Angle-aware Block
# ============================================================
    def __init__(self,hidden_dim):
        super().__init__()
        self.angle_mlp=nn.Sequential(
            nn.Linear(hidden_dim+1,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )

    def forward(self,h,pos,edge_index):
        src,dst=edge_index
        rel=pos[src]-pos[dst]
        rel=F.normalize(rel,p=2,dim=-1)
        agg=torch.zeros_like(h)
        for node in torch.unique(dst):
            mask=(dst==node)
            neigh=src[mask]
            if neigh.numel()<2:
                continue
            vec=rel[mask]
            msg=h[neigh]
            n=vec.size(0)
            for i in range(n):
                for j in range(i+1,n):
                    cos=(vec[i]*vec[j]).sum().unsqueeze(0)
                    a=self.angle_mlp(torch.cat([ 0.5*(msg[i]+msg[j]), cos ], dim=0))
                    agg[node]+=a
        return agg


class EGNNBlock(nn.Module):
# ============================================================
# EGNN Block
# ============================================================
    def __init__(self,hidden_dim,num_rbf=32):
        super().__init__()
        self.edge_encoder=EdgeEncoder(hidden_dim,num_rbf)
        self.edge_mlp=nn.Sequential(
            nn.Linear(hidden_dim*3,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.edge_gate=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.Sigmoid()
        )
        self.node_mlp=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.coord_mlp=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,1)
        )
        self.angle=AngleAwareBlock(hidden_dim)
        self.norm=nn.LayerNorm(hidden_dim)

    def forward(self,h,pos,edge_index,bond_type=None):
        src,dst=edge_index
        rel=pos[src]-pos[dst]
        dist=rel.pow(2).sum(-1).sqrt().unsqueeze(-1)
        edge=self.edge_encoder(dist,bond_type)
        msg=torch.cat(
            [
                h[src],
                h[dst],
                edge
            ],
            dim=-1
        )
        msg=self.edge_mlp(msg)
        gate=self.edge_gate(msg)
        msg=msg*gate
        agg=torch.zeros_like(h)
        agg.index_add_(0,dst,msg)
        agg=agg+self.angle(h,pos,edge_index)
        h=self.norm(h+self.node_mlp(agg))
        direction=rel/(dist+1e-8)
        delta=direction*self.coord_mlp(msg)
        coord=torch.zeros_like(pos)
        coord.index_add_(0,dst,delta)
        pos=pos+coord
        return h,pos



class LayerAttention(nn.Module):
# ============================================================
# Layer Attention
# ============================================================
    def __init__(self,hidden_dim):
        super().__init__()
        self.score=nn.Linear(hidden_dim,1)

    def forward(self,layers):
        stack=torch.stack(layers,dim=1)
        w=self.score(stack)
        w=torch.softmax(w,dim=1)
        return (stack*w).sum(dim=1)
    


class TDAEncoder(nn.Module):
# ============================================================
# TDA Encoder
# ============================================================
    def __init__(self,n_tda=128,hidden_dim=128):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(n_tda,hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim,hidden_dim)
        )

    def forward(self,x):
        return self.net(x)



class Fusion(nn.Module):
# ============================================================
# Fusion Module
# ============================================================
    def __init__(self,hidden_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(hidden_dim*3,hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim,hidden_dim)
        )

    def forward(self,h_geom,h_tda,h_task):
        return self.net(torch.cat([h_geom,h_tda,h_task],dim=-1))



class Backbone(nn.Module):
# ============================================================
# Shared Backbone
# ============================================================
    def __init__(self,hidden_dim,dropout=0.1):
        super().__init__()

        self.block1=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

        self.block2=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self,h):
        h=h+self.block1(h)
        h=h+self.block2(h)
        return h



class EGNNEncoder(nn.Module):
# ============================================================
# EGNN Encoder
# ============================================================
    def __init__(self,hidden_dim=128,layers=6):
        super().__init__()
        self.atom_encoder=AtomEncoder(hidden_dim)
        self.layers=nn.ModuleList([EGNNBlock(hidden_dim) for _ in range(layers)])
        self.layer_attention=LayerAttention(hidden_dim)

    def forward(self, z, pos, edge_index=None, batch=None, charge=None, 
                aromatic=None, edge_attr=None,
    ):
        h=self.atom_encoder(z, charge, aromatic)
        if edge_index is None:
            edge_index=radius_graph(pos, r=5.0, batch=batch, loop=False)

        hidden=[]
        for layer in self.layers:
            h,pos=layer(h, pos, edge_index, edge_attr)
            hidden.append(h)
        h=self.layer_attention(hidden)
        graph=global_mean_pool(h, batch)
        return graph,pos

@MODELS.register(["egnn_v3"])
class MolecularEGNN(BaseModel):
    def __init__(self,hidden_dim=128,num_tasks=10,n_tda=128,layers=6,dropout=0.1):
        super().__init__(num_tasks=num_tasks)
        self.encoder=EGNNEncoder(hidden_dim,layers)
        self.tda=TDAEncoder(n_tda,hidden_dim)
        self.task_emb=nn.Embedding(num_tasks,hidden_dim)
        self.fusion=Fusion(hidden_dim)
        self.backbone=Backbone(hidden_dim,dropout)
        self.energy_head=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
        self.heads=nn.ModuleDict({"y":nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1)),"dipole":nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1)),"polar":nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))})
    
    def forward(self,batch,compute_force=False):
        pos=batch.pos
        if compute_force:
            pos=pos.clone().requires_grad_(True)
        h_geom,pos_out=self.encoder(batch.z,pos,batch.edge_index,batch.batch,getattr(batch,"charge",None),getattr(batch,"aromatic",None),getattr(batch,"edge_attr",None))
        if hasattr(batch,"tda") and batch.tda is not None:
            h_tda=self.tda(batch.tda.float())
        else:
            h_tda=torch.zeros_like(h_geom)
        if hasattr(batch,"task_id"):
            task=self.task_emb(batch.task_id.long().clamp(0,self.task_emb.num_embeddings-1))
        else:
            task=torch.zeros_like(h_geom)
        h=self.fusion(h_geom,h_tda,task)
        h=self.backbone(h)
        out={"repr":h}
        energy=self.energy_head(h)
        out["energy"]=energy
        for name,head in self.heads.items():
            out[name]=head(h)
        if compute_force:
            force=-torch.autograd.grad(energy.sum(),pos,create_graph=True,retain_graph=True)[0]
            out["force"]=force
        return out

# ========================================================
# LOSS FUNCTION
# ========================================================
    def loss_fn(self,outputs,batch):
        loss=torch.tensor(0.,device=outputs["repr"].device)
        if hasattr(batch,"y"):
            pred=outputs["y"];target=batch.y
            if hasattr(batch,"mask_y"):
                mask=batch.mask_y.float()
                loss+=( ((pred-target)**2)*mask ).sum()/mask.sum().clamp(min=1)
            else:
                loss+=F.mse_loss(pred,target)
        if hasattr(batch,"dipole"):
            pred=outputs["dipole"];target=batch.dipole
            if hasattr(batch,"mask_dipole"):
                mask=batch.mask_dipole.float()
                loss+=(torch.abs(pred-target)*mask).sum()/mask.sum().clamp(min=1)
            else:
                loss+=F.l1_loss(pred,target)
        if hasattr(batch,"polar"):
            pred=outputs["polar"];target=batch.polar
            if hasattr(batch,"mask_polar"):
                mask=batch.mask_polar.float()
                loss+=(((pred-target)**2)*mask).sum()/mask.sum().clamp(min=1)
            else:
                loss+=F.mse_loss(pred,target)
        if hasattr(batch,"energy"):
            loss+=F.mse_loss(outputs["energy"],batch.energy)
        if hasattr(batch,"force") and "force" in outputs:
            loss+=F.mse_loss(outputs["force"],batch.force)
        return loss

'''
    def loss_fn(self, outputs, batch):
        loss = 0.0
        eps = 1e-8
# -----------------------------
# TASK 1: y (gap / HOMO-LUMO)
# -----------------------------
        if hasattr(batch, "y"):
            y_true = batch.y
            y_pred = outputs["y"]
            mask = getattr(batch, "mask_y", torch.ones_like(y_true))
            mse = (y_pred - y_true) ** 2
            loss_y = (mse * mask).sum() / (mask.sum().clamp(min=1) + eps)
            loss += loss_y
# -----------------------------
# TASK 2: dipole
# -----------------------------
        if hasattr(batch, "dipole"):
            loss_dip = torch.abs(outputs["dipole"] - batch.dipole).mean()
            loss += loss_dip
# -----------------------------
# TASK 3: polarizability
# -----------------------------
        if hasattr(batch, "polar"):
            loss_polar = ((outputs["polar"] - batch.polar) ** 2).mean()
            loss += loss_polar
        return loss 


    def loss_fn(self, outputs, batch):
#        классический подход из статьи Kendall et al., 2018
        eps = 1e-8
        losses = []
        # -----------------------------
        # TASK 1: y
        # -----------------------------
        if hasattr(batch, "y"):
            y_true = batch.y
            y_pred = outputs["y"]

            mask = getattr(batch, "mask_y", torch.ones_like(y_true))

            mse = (y_pred - y_true).pow(2)
            loss_y = (mse * mask).sum() / (mask.sum().clamp(min=1) + eps)
        else:
            loss_y = None
        # -----------------------------
        # TASK 2: dipole
        # -----------------------------
        if hasattr(batch, "dipole"):
            mask = getattr(batch, "mask_dipole",
                        torch.ones_like(batch.dipole))

            mae = torch.abs(outputs["dipole"] - batch.dipole)
            loss_dip = (mae * mask).sum() / (mask.sum().clamp(min=1) + eps)
        else:
            loss_dip = None
        # -----------------------------
        # TASK 3: polarizability
        # -----------------------------
        if hasattr(batch, "polar"):
            mask = getattr(batch, "mask_polar",
                        torch.ones_like(batch.polar))

            mse = (outputs["polar"] - batch.polar).pow(2)
            loss_polar = (mse * mask).sum() / (mask.sum().clamp(min=1) + eps)
        else:
            loss_polar = None
        # -----------------------------
        # Automatic uncertainty weighting
        # -----------------------------
        total_loss = 0.0

        if loss_y is not None:
            precision = torch.exp(-self.log_vars[0])
            total_loss += precision * loss_y + self.log_vars[0]

        if loss_dip is not None:
            precision = torch.exp(-self.log_vars[1])
            total_loss += precision * loss_dip + self.log_vars[1]

        if loss_polar is not None:
            precision = torch.exp(-self.log_vars[2])
            total_loss += precision * loss_polar + self.log_vars[2]

        return total_loss
    '''