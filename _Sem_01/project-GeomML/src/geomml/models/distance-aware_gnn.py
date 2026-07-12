import torch
import torch.nn as nn
from torch_geometric.nn import radius_graph, global_mean_pool
from geomml.utils.base_model import BaseModel
from geomml.registry import MODELS

class EGNNLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(dim * 2 + 1, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, h, pos, edge_index):
        src, dst = edge_index

        rij = pos[src] - pos[dst]
        dij = (rij ** 2).sum(dim=-1, keepdim=True)

        m = torch.cat([h[src], h[dst], dij], dim=-1)
        m = self.edge_mlp(m)

        agg = torch.zeros_like(h)
        agg.index_add_(0, dst, m)

        h = h + self.node_mlp(agg)
        return h


class EGNNEncoder(nn.Module):
    def __init__(self, dim=128, layers=4, cutoff=4.5):
        super().__init__()
        self.cutoff = cutoff

        self.emb = nn.Embedding(100, dim)
        self.layers = nn.ModuleList([EGNNLayer(dim) for _ in range(layers)])

    def forward(self, z, pos, batch):
        h = self.emb(z.long())

        edge_index = radius_graph(
            pos,
            r=self.cutoff,
            batch=batch,
            loop=False
        )

        for layer in self.layers:
            h = layer(h, pos, edge_index)

        return global_mean_pool(h, batch)

@MODELS.register(["distance-aware_gnn"])    
class MolecularModel(BaseModel):
    '''
    real E(3)-equivariant encoder (EGNN)
    '''
    def __init__(self, num_tasks=3, dim=128, n_tda=128):
        super().__init__()

        self.graph = EGNNEncoder(dim=dim)
        self.tda = nn.Sequential(
            nn.Linear(n_tda, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.task_emb = nn.Embedding(num_tasks, dim)

        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, 1)
        )

    def forward(self, batch):
        h_g = self.graph(batch.z, batch.pos, batch.batch)

        if hasattr(batch, "tda"):
            h_t = self.tda(batch.tda.float())
        else:
            h_t = torch.zeros_like(h_g)

        h_task = self.task_emb(batch.task_id)

        h = torch.cat([h_g, h_t, h_task], dim=-1)
        h = self.fusion(h)

        return self.head(h)
    