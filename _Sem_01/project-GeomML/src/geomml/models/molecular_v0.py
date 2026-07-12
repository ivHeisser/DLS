import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool
from geomml.utils.base_encoder import *
from geomml.registry import MODELS


class SimpleGeomML(nn.Module):
    '''
## 1. Geometry Encoder (EGNN упрощённый)
Если без egnn-pytorch, делаем simplified GeomML:
    '''
    def __init__(self, emb_dim=256):
        super().__init__()

        self.atom_emb = nn.Embedding(100, emb_dim)

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim + 3, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def forward(self, z, pos, batch):
        h = self.atom_emb(z)            # (N,256)
        x = torch.cat([h, pos], dim=-1)
        h = self.mlp(x)
        h = global_mean_pool(h, batch)  # (B,256)
        return h


class TDAEncoder(BaseEncoder):
    '''
## 2. TDA Encoder (заглушка PersLay)
    '''
    def __init__(self, n_features):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            # nn.SiLU(),
            nn.Linear(128, 128)
        )

    @property
    def out_dim(self):
        return 128

    def forward(self, x):
        return self.net(x)


@MODELS.register(["molecular__", "neur_ips__"])
class MolecularModel__(nn.Module):
    """
    Clean multimodal molecular model:
    - geometric graph encoder
    - optional TDA features
    - gated fusion
    """

    def __init__(self, n_features=128):
        super().__init__()

        self.geom = SimpleGeomML(256)
        self.tda = TDAEncoder(n_features)

        # projection to common space
        self.tda_proj = nn.Linear(self.tda.out_dim, 256)

        # fusion
        self.fuse = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 256)
        )

        self.head = nn.Sequential(
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, 1)
        )

    def forward__(self, batch):
        # ---------------------------
        # 1. Graph encoding
        # ---------------------------
        z = batch.z
        pos = batch.pos
        batch_idx = batch.batch

        h_geom = self.geom(z, pos, batch_idx)   # (B, 256)

        # ---------------------------
        # 2. TDA encoding (safe)
        # ---------------------------
        tda = getattr(batch, "tda", None)

        if tda is None:
            h_tda = torch.zeros(
                (h_geom.size(0), 256),
                device=h_geom.device
            )
        else:
            tda = tda.to(h_geom.device).float()

            if tda.dim() == 1:
                tda = tda.unsqueeze(0)

            h_tda = self.tda_proj(self.tda(tda))  # (B, 256)

            # safety check
            if h_tda.size(0) != h_geom.size(0):
                raise RuntimeError(
                    f"Batch mismatch: geom={h_geom.size(0)} tda={h_tda.size(0)}"
                )

        # ---------------------------
        # 3. Gated fusion (better than fake attention)
        # ---------------------------
        h = torch.cat([h_geom, h_tda], dim=-1)  # (B, 512)

        g = self.gate(h)                        # (B, 256)

        h_fused = g * h_geom + (1 - g) * h_tda  # (B, 256)

        # ---------------------------
        # 4. Prediction head
        # ---------------------------
        out = self.head(h_fused)

        return out
    
    def forward(self, batch):
            z = batch.z
            pos = batch.pos
            batch_idx = batch.batch

            h_geom = self.geom(z, pos, batch_idx)

            tda = getattr(batch, "tda", None)

            if tda is None:
                h_tda = torch.zeros((h_geom.size(0), 256), device=h_geom.device)
            else:
                tda = tda.to(h_geom.device).float()
                if tda.dim() == 1:
                    tda = tda.unsqueeze(0)
                h_tda = self.tda_proj(self.tda(tda))

            g = self.gate(torch.cat([h_geom, h_tda], dim=-1))

            h = g * h_geom + (1 - g) * h_tda

            return self.head(h)
        
    '''
 1. настоящую cross-attention (multi-head, token-level)
 2. graph transformer вместо SimpleGeomML
 3. правильный multi-task head (QM9 + Alchemy вместе)
 4. masking missing targets (очень важно для Alchemy)
    
    '''