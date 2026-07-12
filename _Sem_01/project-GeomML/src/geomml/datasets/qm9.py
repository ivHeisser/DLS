import torch
from torch_geometric.datasets import QM9
from torch_geometric.transforms import Compose
from geomml.utils.base_graph import BaseGraphDataset
from geomml.utils.transforms import NormalizeTargets, AddMolecularFeatures


class QM9PropertyDataset(BaseGraphDataset):
    def __init__( self, target_index,
        root="data/qm9",
        normalize=False,
        stats=None,
    ):
        self.dataset = QM9(root)
        assert len(self.dataset) > 0
        super().__init__(self.dataset, target_index=target_index)
        transforms = [AddMolecularFeatures()]
        if normalize:
            if stats is None:
                stats = self.compute_stats()
            self.stats = stats
            transforms.append(
                NormalizeTargets(stats, keys=["y"])
            )
        self.set_transform(Compose(transforms))

    def compute_stats(self):
        values = []
        for data in self.dataset:
            y = data.y.float()
            if self.target_index is not None:
                if y.ndim == 2:
                    y = y[:, self.target_index]
                else:
                    y = y[self.target_index]
            values.append( y.view(-1,1) )
        y = torch.cat(values, dim=0)
        return {
            "y_mean": y.mean(dim=0),
            "y_std": y.std(dim=0)
        }

# ===== Factories =====
from geomml.registry import DATASETS

@DATASETS.register("qm9")
def qm9_builder(target_index=4, root="data/qm9", normalize=False, stats=None):
    return QM9PropertyDataset(target_index=target_index, root=root, normalize=normalize, stats=stats)


@DATASETS.register("gap")
def gap_builder(root="data/qm9", normalize=False, stats=None):
    return QM9PropertyDataset(target_index=4, root=root, normalize=normalize, stats=stats)


@DATASETS.register("homo")
def homo_builder(root="data/qm9", normalize=False, stats=None):
    return QM9PropertyDataset(target_index=2, root=root, normalize=normalize, stats=stats)


@DATASETS.register("lumo")
def lumo_builder(root="data/qm9", normalize=False, stats=None):
    return QM9PropertyDataset(target_index=3, root=root, normalize=normalize, stats=stats)