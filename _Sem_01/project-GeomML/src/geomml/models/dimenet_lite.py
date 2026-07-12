import torch,torch.nn as nn,torch.nn.functional as F
from torch_cluster import radius_graph
from torch_geometric.nn import global_mean_pool

from geomml.registry import MODELS
from geomml.utils.base_model import BaseModel


class RBF(nn.Module):
# ============================================================
# Radial Basis Expansion
# ============================================================
    def __init__(self,n=32,cutoff=5.): 
        super().__init__()
        self.cutoff=cutoff
        self.gamma=10.
        self.register_buffer("c",torch.linspace(0,cutoff,n))
    def forward(self,d): 
        return torch.exp(-self.gamma*(d-self.c)**2)

class DimeLayer(nn.Module):
# ============================================================
# DimeNet-lite Layer
# ============================================================
    def __init__(self,h=256,nrbf=32):
        super().__init__()
        self.rbf=nn.Sequential(
            nn.Linear(nrbf,h),
            nn.SiLU(),
            nn.Linear(h,h)
        )
        self.msg=nn.Sequential(
            nn.Linear(h*3,h),
            nn.SiLU(),
            nn.Linear(h,h)
        )
        self.upd=nn.Sequential(
            nn.Linear(h,h),
            nn.SiLU(),
            nn.Linear(h,h)
        )
        self.norm=nn.LayerNorm(h)

    def forward(self,x,edge_index,edge_attr):
        s,d=edge_index
        m=self.msg(torch.cat([x[s],x[d],self.rbf(edge_attr)],-1))
        agg=torch.zeros_like(x)
        agg.index_add_(0,d,m)
        return self.norm(x+self.upd(agg))

class DimeEncoder(nn.Module):
# ============================================================
# Encoder
# ============================================================
    def __init__(self,h=256,layers=6,cutoff=5.,nrbf=32):
        super().__init__()
        self.cutoff=cutoff
        self.emb=nn.Embedding(100,h)
        self.rbf=RBF(nrbf,cutoff)
        self.layers=nn.ModuleList([DimeLayer(h,nrbf) for _ in range(layers)])

    def forward(self,z,pos,batch,edge_index=None):
        x=self.emb(z.long())
        if edge_index is None or edge_index.numel()==0: 
            edge_index=radius_graph(pos,r=self.cutoff,batch=batch,loop=False)
        s,d=edge_index
        dist=(pos[s]-pos[d]).norm(dim=-1,keepdim=True)
        e=self.rbf(dist)
        for l in self.layers: 
            x=l(x,edge_index,e)
        '''
Примечание:
В конце global_mean_pool(x,batch) не всегда лучший вариант для молекулярных свойств.
Часто лучше работают/попробовать в будущем:
* Set2Set
* Attention Pool
* Global Add Pool
* Weighted Pool

Особенно если свойства экстенсивные.
        '''
        return global_mean_pool(x,batch)

@MODELS.register(["dimenet_lite"])
class MolecularDimeNetLite(BaseModel):
    def __init__(self,hidden_dim=256,n_tda=128,layers=6,num_tasks=16):
        super().__init__()
        self.geom=DimeEncoder(hidden_dim,layers)
        self.tda=nn.Sequential(
            nn.Linear(n_tda,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.tda_null=nn.Parameter(torch.zeros(hidden_dim))
        self.task=nn.Embedding(num_tasks,hidden_dim)
        self.fusion=nn.Sequential(
            nn.Linear(hidden_dim*3,hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.backbone=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),nn.LayerNorm(hidden_dim)
        )
        self.head=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,1)
        )
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self,batch):
        hg=self.geom(batch.z,batch.pos,batch.batch,getattr(batch,"edge_index",None))
        ht=self.tda(batch.tda.float()) if hasattr(batch,"tda") else self.tda_null.unsqueeze(0).expand(hg.size(0),-1)
        h=torch.cat([hg,ht,self.task(batch.task_id.long())],-1)
        h=self.backbone(self.fusion(h))
        return {"repr":h,"y":self.head(h)}
    
    def loss_fn(self,pred,batch):
        return F.mse_loss(pred["y"],batch.y)

