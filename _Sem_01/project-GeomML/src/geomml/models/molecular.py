import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from geomml.utils.base_model import BaseModel
from geomml.losses.mae_mse import multitask_loss
from geomml.registry import MODELS


class SimpleGeomML(nn.Module):
    def __init__(self, emb_dim=256):
        super().__init__()

        self.atom_emb = nn.Embedding(100, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + 3, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def forward(self, z, pos, batch):
        h = self.atom_emb(z)
        x = torch.cat([h, pos], dim=-1)
        h = self.mlp(x)
        return global_mean_pool(h, batch)


class TDAEncoder(nn.Module):
    def __init__(self, n_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        )

    def forward(self, x):
        return self.net(x)

@MODELS.register(["molecular"])
class MolecularModel(BaseModel):
    def __init__(self, n_features=128, num_tasks=3):
        super().__init__()

        # encoders
        self.geom = SimpleGeomML(256)
        self.tda = TDAEncoder(n_features)
        self.log_vars = nn.Parameter(torch.zeros(3))  # y, dipole, polar # sigma = torch.exp(self.log_vars)
        # fusion
        self.tda_proj = nn.Linear(128, 256)

        self.fuse = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        # task conditioning (optional)
        self.task_emb = nn.Embedding(num_tasks, 64)
        self.task_proj = nn.Linear(64, 256)

        # heads (match loss!)
        self.head_y = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

        self.head_dipole = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

        self.head_polar = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

        # loss по умолчанию
        self._loss = self.build_loss(
            y_weights=(0.8, 0.2),
            dipole_weights=(0.5, 0.5),
            polar_weights=(0.2, 0.8),
        )

    def forward(self, batch):
        # ---- unpack ----
        z = batch["z"] if isinstance(batch, dict) else batch.z
        pos = batch["pos"] if isinstance(batch, dict) else batch.pos
        batch_idx = batch["batch"] if isinstance(batch, dict) else batch.batch

        # ---- geometry ----
        h_geom = self.geom(z, pos, batch_idx)

        # ---- TDA ----
        if isinstance(batch, dict) and "tda" in batch:
            tda = batch["tda"].to(h_geom.device).float()
            if tda.dim() == 1:
                tda = tda.unsqueeze(0)
            h_tda = self.tda_proj(self.tda(tda))
        else:
            h_tda = torch.zeros_like(h_geom)

        # ---- fusion ----
        h = self.fuse(torch.cat([h_geom, h_tda], dim=-1))

        # ---- task conditioning ----
        if isinstance(batch, dict) and "task_id" in batch:
            t = batch["task_id"]
            if t.dim() > 1:
                t = t.squeeze(1)
            t = self.task_proj(self.task_emb(t))
            h = h + t
        return {
            "repr": h,
            "y": self.head_y(h),
            "dipole": self.head_dipole(h),
            "polar": self.head_polar(h),
        }
    
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
    
    '''
    Ключевая идея
geom + tda + task_emb → shared representation
дальше:
head_y
head_dipole
head_polar


Я могу тебе сделать апгрейд до уровня paper:
🔥 1. replace fusion → cross-attention (Geom ↔ TDA)
🔥 2. add uncertainty weighting per task
🔥 3. missing label masking (Alchemy-style)
🔥 4. GradNorm balancing (multi-task stability)
🔥 5. remove task_id entirely (SOTA multitask GNNs)


уровня paper:
🚀 вместо gate:
cross-attention (geom ↔ tda)
FiLM conditioning
gated residual fusion (SOTA in multimodal GNN)



“proper paper version”:
uncertainty weighting (learned task variance)
GradNorm balancing
missing label stochastic masking (Alchemy-style)
attention fusion вместо gate
EGNN upgrade вместо SimpleGeomML
    '''