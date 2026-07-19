import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from geomml.utils.base_model import BaseModel
from geomml.losses.mae_mse import multitask_loss
from geomml.registry import MODELS
from torch_geometric.nn import (
    GINConv, GINEConv,
    global_mean_pool,
)

# ---------------------------------------------------------
# Graph Encoder (geometry-only)
# ---------------------------------------------------------
class GraphEncoder(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.atom_emb = nn.Embedding(100, hidden)
        self.coord_mlp = nn.Sequential(
            nn.Linear(3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, z, pos, batch):
        z = z.long()
        h_atom = self.atom_emb(z)
        h_pos = self.coord_mlp(pos)
        h = torch.cat([h_atom, h_pos], dim=-1)
        h = self.fuse(h)
        h = self.norm(h)
        return global_mean_pool(h, batch)


# ---------------------------------------------------------
# TDA Encoder
# ---------------------------------------------------------
class TDAEncoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 256),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# Fusion (stable gate)
# ---------------------------------------------------------
class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid()
        )

    def forward(self, g, t):
        x = torch.cat([g, t], dim=-1)
        a = self.gate(x)
        return a * g + (1 - a) * t


# ---------------------------------------------------------
# Residual block
# ---------------------------------------------------------
class Residual(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


# ---------------------------------------------------------
# Heads (always return Tensor)
# Улучшение в heads: 
# Linear → SiLU → LayerNorm → Linear → 1
#               или
# Linear → SiLU → Dropout → Linear
#           вместо
# Linear → SiLU → Linear → 1
# ---------------------------------------------------------
class RegressionHead_1(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        return self.net(x)


class RegressionHead_2(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------
# MAIN MODEL (TRAINER-COMPATIBLE)
# ---------------------------------------------------------
@MODELS.register(["molecular_v1"])
class MolecularModel(BaseModel):
    def __init__(self, n_features=128, num_tasks=3):
        super().__init__(num_tasks=num_tasks)
        self.log_vars = nn.Parameter(torch.zeros(3)) 
        self.graph = GraphEncoder()
        self.tda = TDAEncoder(n_features)
        self.tda_null = nn.Parameter(torch.zeros(256))
        self.fusion = Fusion()

        self.task_emb = nn.Embedding(num_tasks, 64)

        self.project = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
        )

        self.res = nn.Sequential(
            Residual(),
            Residual(),
        )

        # match dataset keys exactly
        self.heads = nn.ModuleDict({
            "y": RegressionHead_1(),
            "dipole": RegressionHead_1(),
            "polar": RegressionHead_2(),
#            "qm9": RegressionHead(),
#            "esol": RegressionHead(),
#            "molhiv": ClassificationHead(),
        })

    def forward(self, batch):
        # -----------------------------
        # SAFE extraction
        # -----------------------------
        z = batch["z"]
        pos = batch["pos"]
        b = batch["batch"]

        # -----------------------------
        # graph encoding
        # -----------------------------
        h_g = self.graph(z, pos, b)

        # -----------------------------
        # optional TDA
        # -----------------------------
        if ("tda" in batch) and (batch["tda"] is not None):
            h_t = self.tda(batch["tda"].float())
        else:
            h_t = self.tda_null.unsqueeze(0).expand(h_g.size(0), -1)

        # -----------------------------
        # fusion
        # -----------------------------
        h = self.fusion(h_g, h_t)

        # -----------------------------
        # task embedding
        # -----------------------------
        task = self.task_emb(batch["task_id"]).squeeze(1)

        h = torch.cat([h, task], dim=-1)
        h = self.project(h)
        h = self.res(h)

        # -----------------------------
        # outputs (STRICT tensor dict)
        # -----------------------------
        out = {"repr": h,}
        for k, head in self.heads.items():
            out[k] = head(h)
        
        return out
    
    def loss_fn(self, outputs, batch):
        '''
        Метод loss_fn — обычный метод экземпляра.
        Плюсы:
        + не нужно подбирать веса
        + автоматически балансирует шкалы loss’ов
        + почти не ломается
        + очень стабилен в PyG / molecular tasks
        '''
        loss = 0.
        if hasattr(batch, "y"):
            mask = getattr(batch, "mask_y", torch.ones_like(batch.y))
            loss += ((outputs["y"] - batch.y) ** 2 * mask).sum() / mask.sum()
        if hasattr(batch, "dipole"):
            mask = getattr(batch, "mask_dipole", torch.ones_like(batch.dipole))
            loss += (torch.abs(outputs["dipole"] - batch.dipole) * mask).mean()
        if hasattr(batch, "polar"):
            mask = getattr(batch, "mask_polar", torch.ones_like(batch.polar))
            loss += ((outputs["polar"] - batch.polar) ** 2 * mask).sum() / mask.sum()
        return loss
        #return self._loss(outputs, batch, self)
    
    @staticmethod
    def build_loss(
        y_weights=(0.8, 0.2),
        dipole_weights=(0.5, 0.5),
        polar_weights=(0.2, 0.8),
    ):
        '''
    Фабрика функции потерь — статическая или классовая.
        '''
        return multitask_loss(
            y_weights=y_weights,
            dipole_weights=dipole_weights,
            polar_weights=polar_weights,
        )