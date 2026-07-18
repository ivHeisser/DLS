import torch
import torch.nn as nn
import torch.nn.functional as F
from geomml.registry import MODELS
from geomml.utils.base_model import BaseModel
from geomml.models.dimenet_lite import DimeEncoder


@MODELS.register(["qm9s_dimenet"])
class QM9SDimeNet(BaseModel):

    def __init__( self, hidden_dim=256, n_tda=128, layers=6 ):
        super().__init__()
        self.geom=DimeEncoder( h=hidden_dim, layers=layers )
        self.tda=nn.Sequential(
            nn.Linear(n_tda,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,hidden_dim)
        )
        self.tda_null=nn.Parameter( torch.zeros(hidden_dim) )
        self.fusion=nn.Sequential(
            nn.Linear(hidden_dim*2,hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.backbone=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim)
        )
        self.dipole_head=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,3)
        )
        self.polar_head=nn.Sequential(
            nn.Linear(hidden_dim,hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim,9)
        )
        self.log_vars=nn.Parameter(torch.zeros(2))

    def forward(self,batch):
        hg=self.geom( batch.z, batch.pos, batch.batch, getattr(batch,"edge_index",None) )
        if hasattr(batch,"tda"):
            ht=self.tda(batch.tda.float())
        else:
            ht=self.tda_null.unsqueeze(0).expand( hg.size(0),-1 )

        h=torch.cat( [hg,ht], dim=-1 )
        h=self.fusion(h)
        h=self.backbone(h)
        return {
            "repr":h,
            "dipole":self.dipole_head(h),
            "polar":self.polar_head(h)
        }

    def loss_fn(self,pred,batch):
        dipole_target=batch.dipole.view(pred["dipole"].shape)
        polar_target=batch.polar.view(pred["polar"].shape)
        loss_d=F.mse_loss(pred["dipole"],dipole_target)
        loss_p=F.mse_loss(pred["polar"],polar_target)
        loss=torch.exp(-self.log_vars[0])*loss_d+self.log_vars[0]+torch.exp(-self.log_vars[1])*loss_p+self.log_vars[1]
        return loss