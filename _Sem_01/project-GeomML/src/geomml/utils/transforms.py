
from torch_geometric.transforms import BaseTransform
from torch_cluster import radius_graph
import torch


class NormalizeTargets(BaseTransform):
    def __init__(self, stats, keys):
        self.stats = stats
        self.keys = keys

    def forward(self,data):
        eps = 1e-8
        for key in self.keys:
            if hasattr(data, f"mask_{key}"):
                mask = getattr(data, f"mask_{key}")
                x[mask == 0] = 0
                mean = self.stats[f"{key}_mean"]
                std = self.stats[f"{key}_std"]
                x = (x - mean) / (std + eps)
                setattr(data, key, x)
        return data
    


class AddMolecularFeatures(BaseTransform):
    def __init__(self, cutoff=1.8):
        self.cutoff=cutoff

    def forward(self,data):
        n=data.z.size(0)
# --------------------------------
# formal charge
# QM9 neutral molecules
# --------------------------------
        if not hasattr(data,"charge"):
            data.charge=torch.zeros( n, dtype=torch.long )
# --------------------------------
# aromatic
# QM9 does not provide it
# default = 0
# --------------------------------
        if not hasattr(data,"aromatic"):
            data.aromatic=torch.zeros( n, dtype=torch.long )

        if not hasattr(data,"edge_attr"):
            data.edge_attr=torch.zeros( data.edge_index.size(1), 1, dtype=torch.long )
# --------------------------------
# create bonds
# --------------------------------
        edge_index=radius_graph( data.pos, r=self.cutoff, loop=False )
        src,dst=edge_index
        distances=( data.pos[src] - data.pos[dst] ).norm( dim=-1 )
# crude bond type
#
# 0 unknown
# 1 single
# 2 double
# 3 triple
        bond=torch.zeros(distances.size(0), dtype=torch.long)
        bond[distances < 1.25]=2
        bond[distances < 1.15]=3
        data.edge_index=edge_index
        data.edge_attr=bond
        return data
    
class NormalizeByTask:
    def __init__(self, task_stats, keys=("y", "dipole", "polar")):
        self.task_stats = task_stats
        self.keys = keys

    def __call__(self, data):
        task = int(data.task_id)

        stats = self.task_stats[task]

        for key in self.keys:
            if not hasattr(data, key):
                continue

            mean_key = f"{key}_mean"
            std_key = f"{key}_std"

            if mean_key not in stats:
                continue

            x = getattr(data, key)

            if hasattr(data, f"mask_{key}"):
                mask = getattr(data, f"mask_{key}").bool()
                x = x.clone()
                x[mask] = (x[mask] - stats[mean_key]) / stats[std_key]
            else:
                x = (x - stats[mean_key]) / stats[std_key]

            setattr(data, key, x)

        return data