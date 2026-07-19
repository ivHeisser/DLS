import torch
import torch.nn as nn
from geomml.utils.base_model import BaseModel

from torch_geometric.nn import (
    GINEConv,
    global_mean_pool,
)

from geomml.registry import MODELS


# ============================================================
# Graph Encoder
# ============================================================

class GraphEncoder(nn.Module):
    def __init__(self, hidden_dim=256, num_layers=4):
        super().__init__()
        self.atom_emb = nn.Embedding(100, hidden_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.convs = nn.ModuleList()

        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.convs.append(
                GINEConv(
                    nn=mlp,
                    edge_dim=hidden_dim,
                )
            )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, z, pos, edge_index, batch):
        h = self.atom_emb(z.long())
        row, col = edge_index
        edge_attr = pos[row] - pos[col]
        edge_attr = self.edge_mlp(edge_attr)
        for conv in self.convs:
            h = h + conv(h, edge_index, edge_attr)
        h = self.norm(h)
        return global_mean_pool(h, batch)


# ============================================================
# TDA Encoder
# ============================================================

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


# ============================================================
# Fusion
# ============================================================

class Fusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(512, 256),
            nn.Sigmoid(),
        )

    def forward(self, geom, tda):
        g = self.gate(torch.cat([geom, tda], dim=-1))
        return g * geom + (1 - g) * tda


# ============================================================
# Residual MLP
# ============================================================

class ResidualBlock(nn.Module):
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


# ============================================================
# Main model
# ============================================================

@MODELS.register(["molecular_v2"])
class MolecularModel(BaseModel):
    def __init__(
        self,
        n_features=128,
        num_tasks=3,
    ):
        super().__init__(num_tasks=num_tasks)
        self.log_vars = nn.Parameter(torch.zeros(3)) 
        self.geom = GraphEncoder()
        self.tda = TDAEncoder(n_features)
        self.fusion = Fusion()
        self.task_emb = nn.Embedding(num_tasks, 64)
        self.proj = nn.Sequential(
            nn.Linear(256 + 64, 256),
            nn.SiLU(),
            nn.LayerNorm(256),
        )
        self.backbone = nn.Sequential(
            ResidualBlock(),
            ResidualBlock(),
        )
        self.heads = nn.ModuleDict({
            "y": nn.Sequential(
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            ),
            "dipole": nn.Sequential(
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            ),
            "polar": nn.Sequential(
                nn.Linear(256, 256),
                nn.SiLU(),
                nn.Linear(256, 1),
            ),
        })

    def forward(self, batch):
        z = batch["z"]
        pos = batch["pos"]
        edge_index = batch["y"] # batch["edge_index"]
        batch_idx = batch["batch"]
        h_geom = self.geom(
            z,
            pos,
            edge_index,
            batch_idx,
        )
        if "tda" in batch and batch["tda"] is not None:
            h_tda = self.tda(batch["tda"].float())
        else:
            h_tda = torch.zeros_like(h_geom)
        h = self.fusion(h_geom, h_tda)
        task = self.task_emb(
            batch["task_id"]
        ).squeeze(1)
        h = torch.cat([h, task], dim=-1)
        h = self.proj(h)
        h = self.backbone(h)
        out = {"repr": h,}
        for name, head in self.heads.items():
            out[name] = head(h)

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

    def get_model_name(self):
        return "molecular_v2"
    
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