import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_scatter import scatter_mean, scatter_softmax
from torch_geometric.nn import global_mean_pool
from torch_cluster import radius_graph

from geomml.utils.base_model import BaseModel
from geomml.registry import MODELS

# ============================================================
# Radial Basis Expansion
# ============================================================
class GaussianRBF(nn.Module):
    def __init__(self, cutoff=5.0, n_rbf=32):
        super().__init__()
        centers = torch.linspace(0, cutoff, n_rbf)
        self.register_buffer("centers", centers)
        self.gamma = (10.0 /  cutoff)

    def forward(self, dist):
        diff = (dist - self.centers)
        return torch.exp(-self.gamma * diff**2 )

# ============================================================
# Edge Attention
# ============================================================
class EdgeAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, m, dst):
        a = self.score(m)
        a = scatter_softmax(a, dst)
        return m * a

# ============================================================
# EGNN Layer
# ============================================================
class AdvancedEGNNLayer(nn.Module):
    def __init__(self, hidden_dim, n_rbf=32, n_bonds=8):
        super().__init__()
        self.rbf = GaussianRBF(
            cutoff=5.0,
            n_rbf=n_rbf
        )
        self.bond_emb = nn.Embedding(
            n_bonds,
            hidden_dim
        )
        self.edge_mlp = nn.Sequential( # edge message
            nn.Linear( hidden_dim*3+n_rbf,  hidden_dim),
            nn.SiLU(),
            nn.Linear( hidden_dim, hidden_dim)
        )
        self.attn = EdgeAttention( hidden_dim )
        self.node_norm = nn.LayerNorm( hidden_dim ) # node update
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.coord_mlp = nn.Sequential( # coordinate update
            nn.Linear(hidden_dim, 1),
            nn.Tanh()
        )

    def forward(self, h, pos, edge_index, bond_type):
        src,dst=edge_index
        rel = ( pos[src] - pos[dst] )
        dist = torch.norm(rel, dim=-1, keepdim=True)
        rbf = self.rbf(dist)
        bond = self.bond_emb(bond_type)
        edge_input=torch.cat([h[src], h[dst], bond, rbf],  dim=-1)
        m=self.edge_mlp(edge_input)
        m=self.attn(m, dst)                                     # attention
        agg=scatter_mean( m, dst, dim=0, dim_size=h.size(0))    # node aggregation
        h=h+self.node_mlp(self.node_norm(agg))
        scale=self.coord_mlp(m)                                 # coordinate update
        delta=rel*scale*0.1
        delta=scatter_mean(delta, dst, dim=0, dim_size=pos.size(0))
        pos=pos+delta
        return h,pos


class AtomEncoder(nn.Module):
# ============================================================
# Atom Feature Encoder
# ============================================================
    def __init__(self, hidden_dim,
        max_atomic_num=128,
        max_charge=16,
    ):
        super().__init__()
        self.z_emb = nn.Embedding(
            max_atomic_num,
            hidden_dim
        )
        self.charge_emb = nn.Embedding(
            max_charge,
            hidden_dim
        )
        self.aromatic_emb = nn.Embedding(
            2,
            hidden_dim
        )
        self.out = nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )

    def forward(self, z, charge, aromatic):
        h=torch.cat(
            [
                self.z_emb(z),
                self.charge_emb(charge),
                self.aromatic_emb(aromatic)
            ],
            dim=-1
        )
        return self.out(h)


class LayerAttention(nn.Module):
# ============================================================
# Layer Attention: Learns which EGNN depth is important
# ============================================================
    def __init__(self, dim):
        super().__init__()
        self.score=nn.Linear(dim, 1)

    def forward(self, layers): # layers: [N_layers,B,dim] 
        x=torch.stack(layers, dim=0)
        w=self.score(x)
        w=torch.softmax(w, dim=0)
        return (x*w).sum(dim=0)


class AdvancedEGNNEncoder(nn.Module):
# ============================================================
# EGNN Encoder
# ============================================================
    def __init__(self, hidden_dim=256, layers=6, cutoff=5.0):
        super().__init__()
        self.atom_encoder=AtomEncoder(
            hidden_dim
        )
        self.layers=nn.ModuleList(
            [ AdvancedEGNNLayer(hidden_dim)
                for _ in range(layers)
            ]
        )
        self.layer_attention=LayerAttention(
            hidden_dim
        )
        self.cutoff=cutoff

    def forward(self, z, charge, aromatic, pos, batch,
        edge_attr=None
    ):
        h=self.atom_encoder(z, charge, aromatic)
        outputs=[]
        for layer in self.layers: 
            edge_index=radius_graph(pos,
                r=self.cutoff,
                batch=batch,
                loop=False
            ) # rebuild graph after coordinate update

            src,dst=edge_index
            if edge_attr is None: # if no bonds provided assume single bond
                bond_type=torch.zeros(
                    src.shape[0],
                    device=pos.device,
                    dtype=torch.long
                )
            else:
                bond_type=edge_attr.long()
            h,pos=layer(h, pos, edge_index, bond_type)
            graph_h=global_mean_pool(h, batch)
            outputs.append(graph_h)

        graph_repr=self.layer_attention(outputs)
        return graph_repr,pos


@MODELS.register(["egnn_advanced"])
class MolecularEGNN_adv(BaseModel):
# ============================================================
# Advanced Molecular EGNN
# ============================================================
    def __init__(self, hidden_dim=256, n_tasks=32, n_tda=128):
        super().__init__(num_tasks=4)

        self.encoder=AdvancedEGNNEncoder(hidden_dim)

        self.tda=nn.Sequential(                     # TDA branch
            nn.Linear(n_tda, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.task_emb=nn.Embedding(n_tasks, hidden_dim)

        self.fusion=nn.Sequential(
            nn.Linear(hidden_dim*3, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.energy_head=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        self.dipole_head=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3)
        )

        self.polar_head=nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )


    def forward(self, batch, compute_force=True):
    # ========================================================
    # Forward
    # ========================================================
        pos=batch.pos
        if compute_force:
            pos.requires_grad_(True)
        h,pos_out=self.encoder(batch.z, batch.charge, batch.aromatic, pos, 
                               batch.batch, getattr(batch, "edge_attr", None)
        )

        if hasattr(batch,"tda"):
            tda=self.tda(batch.tda.float())
        else:
            tda=torch.zeros_like(h)

        task=self.task_emb(batch.task_id)
        h=self.fusion(torch.cat( [ h, tda,  task ], dim=-1 ))
        energy=self.energy_head(h)
        result={
            "energy":energy,
            "repr":h,
            "dipole": self.dipole_head(h),
            "polar": self.polar_head(h)
        }

        if compute_force:                 # Forces
            force=torch.autograd.grad(
                energy.sum(),
                batch.pos,
                create_graph=True
            )[0]
            result["forces"]=-force
        return result

# ============================================================
# Loss
# ============================================================
    def loss_fn(self, output, batch):
        losses=[]
        total=0

        if hasattr(batch,"y"):                           # energy
            losses.append(
                F.mse_loss(output["energy"], batch.y)
            )
        if hasattr(batch,"forces"):                      # force
            losses.append(
                F.mse_loss(
                    output["forces"],
                    batch.forces
                )
            )
        if hasattr(batch,"dipole"):                     # dipole
            losses.append(
                F.l1_loss(
                    output["dipole"],
                    batch.dipole
                )
            )
        if hasattr(batch,"polar"):                       # polar
            losses.append(
                F.mse_loss(
                    output["polar"],
                    batch.polar
                )
            )
        for i,l in enumerate(losses):
            precision=torch.exp(
                -self.log_vars[i]
            )
            total += (precision*l + self.log_vars[i])
        return total    
