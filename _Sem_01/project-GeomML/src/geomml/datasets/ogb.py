import torch
from ogb.graphproppred import PygGraphPropPredDataset
from geomml.utils.graph_adapter import AutoGraphAdapter
from geomml.utils.base_graph import BaseGraphDataset
from geomml.utils.transforms import NormalizeTargets


class OGBGraphPropertyDataset(BaseGraphDataset):
    """
    Wrapper for OGB graph property prediction datasets:
    - ogbg-molhiv
    - ogbg-molpcba
    - ogbg-moltox21
    - etc.
    """
    def __init__(self, name,
        root="data/ogb",
        task_id=0,
        normalize=False,
        stats=None,
    ):
        dataset = PygGraphPropPredDataset(name=name, root=root)
        adapter = AutoGraphAdapter(dataset)
        super().__init__(adapter, task_id=task_id)

        if normalize:
            if stats is None:
                stats = self.compute_stats()
            self.stats = stats
            self.set_transform(
                NormalizeTargets(stats, keys=["y"])
            )

    # -----------------------------------------
    # Compute target statistics
    # -----------------------------------------
    def compute_stats(self):
        old_transform = self.transform
        self.transform = None
        values = []
        for idx in range(len(self)):
            data = self[idx]
            if hasattr(data, "y"):
                values.append(
                    data.y.float()
                )
        y = torch.cat(values, dim=0)
        self.transform = old_transform
        return {
            "y_mean": y.mean(dim=0),
            "y_std": y.std(dim=0)
        }
    
# ===== Factories =====
from geomml.registry import DATASETS

@DATASETS.register("ogbg-molhiv")
def molhiv_builder(root="data/ogb", normalize=False, stats=None):
    return OGBGraphPropertyDataset(
        name="ogbg-molhiv",
        root=root,
        task_id=0,
        normalize=normalize,
        stats=stats
    )

@DATASETS.register("ogbg-molpcba")
def molpcba_builder(root="data/ogb", normalize=False, stats=None):
    return OGBGraphPropertyDataset(
        name="ogbg-molpcba",
        root=root,
        task_id=1,
        normalize=normalize,
        stats=stats
    )

@DATASETS.register("ogbg-moltox21")
def moltox21_builder(root="data/ogb", normalize=False, stats=None):
    return OGBGraphPropertyDataset(
        name="ogbg-moltox21",
        root=root,
        task_id=2,
        normalize=normalize,
        stats=stats
    )