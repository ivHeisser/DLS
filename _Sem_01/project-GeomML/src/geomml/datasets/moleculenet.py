import torch
from torch_geometric.datasets import MoleculeNet
from geomml.utils.base_graph import BaseGraphDataset
from geomml.utils.transforms import NormalizeTargets


class MoleculeNetPropertyDataset(BaseGraphDataset):
    def __init__(self, name,
        target_index=None,
        root="data/moleculenet",
        normalize=False,
        stats=None,
    ):
        dataset = MoleculeNet(root=root, name=name)
        super().__init__(dataset, target_index=target_index)
        if normalize:
            if stats is None:
                stats = self.compute_stats()
            self.stats = stats
            self.set_transform(
                NormalizeTargets(
                    stats,
                    keys=["y"]
                )
            )

    # ---------------------------------
    # Compute target statistics
    # ---------------------------------
    def compute_stats(self):
        old_transform = self.transform
        self.transform = None
        values = []
        for idx in range(len(self)):
            data = self[idx]
            if hasattr(data, "y"):
                y = data.y.float()
                values.append(y)
        y = torch.cat(values, dim=0)
        self.transform = old_transform
        return {
            "y_mean": y.mean(dim=0),
            "y_std": y.std(dim=0)
        }


# ===== Factories =====
from geomml.registry import DATASETS

@DATASETS.register("esol")
def esol_builder(root="data/moleculenet"):
    return MoleculeNetPropertyDataset(
        name="ESOL",
        target_index=0,
        root=root
    )


@DATASETS.register("lipo")
def lipo_builder(root="data/moleculenet"):
    return MoleculeNetPropertyDataset(
        name="Lipophilicity",
        target_index=0,
        root=root
    )


@DATASETS.register("bbbp")
def bbbp_builder(root="data/moleculenet"):
    return MoleculeNetPropertyDataset(
        name="BBBP",
        target_index=0,
        root=root
    )


@DATASETS.register("hiv")
def hiv_builder(root="data/moleculenet"):
    return MoleculeNetPropertyDataset(
        name="HIV",
        target_index=0,
        root=root
    )


@DATASETS.register("tox21")
def tox21_builder(root="data/moleculenet"):
    return MoleculeNetPropertyDataset(
        name="Tox21",
        target_index=None,
        root=root
    )