import torch.nn as nn
from egnn_pytorch import EGNN_Network
from torch_geometric.utils import to_dense_batch
import torch.nn.functional as F
from geomml.utils.base_model import BaseModel
from geomml.registry import MODELS


class EGNNEncoder(nn.Module):
    """
    Input:
        batch.z      : (num_atoms,)     # (B, N)
        batch.pos    : (num_atoms, 3)   # (B, N, 3)
        batch.batch  : (num_atoms,)     # (B, N)

    Output:
        graph embedding: (B, D)
    """

    def __init__(
        self,
        num_atom_types: int = 100,
        emb_dim: int = 128,
        depth: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_atom_types, emb_dim)
        self.egnn = EGNN_Network(
            dim=emb_dim,
            depth=depth,
            num_nearest_neighbors=0,
            dropout=dropout,
        )

    def forward(self, batch):
        # ---------- node features ----------
        z = batch.z.long()
        pos = batch.pos.float()
        batch_idx = batch.batch
        x = self.embedding(z)
        # ---------- convert PyG Batch -> dense ----------
        x, mask = to_dense_batch(x, batch_idx)
        pos, _ = to_dense_batch(pos, batch_idx)
        # ---------- EGNN ----------
        x, _ = self.egnn(
            feats=x,
            coors=pos,
            mask=mask,
        )
        # ---------- masked mean pooling ----------
        mask = mask.unsqueeze(-1).float()
        graph_emb = (x * mask).sum(dim=1)
        graph_emb /= mask.sum(dim=1).clamp(min=1.0)
        return graph_emb


@MODELS.register(["egnn_gap"])
class EGNNGapRegressor(BaseModel):
    def __init__(
        self,
        num_atom_types: int = 100,
        emb_dim: int = 128,
        depth: int = 4,
        mlp_hidden: int = 256,
    ):
        super().__init__(num_tasks=1)
        self.encoder = EGNNEncoder(
            num_atom_types=num_atom_types,
            emb_dim=emb_dim,
            depth=depth,
        )
        self.head = nn.Sequential(
            nn.Linear(emb_dim, mlp_hidden),
            nn.SiLU(),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, batch):
        h = self.encoder(batch)
        pred = self.head(h)
        return {
            "y": pred
        }
    
    def loss_fn(self, pred, batch):
        return F.mse_loss(pred["y"], batch.y)