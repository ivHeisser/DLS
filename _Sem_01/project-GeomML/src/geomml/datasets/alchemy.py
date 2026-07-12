import torch
from torch_geometric.transforms import Compose
from geomml.registry import DATASETS
from geomml.utils.base_graph import BaseGraphDataset
from geomml.utils.transforms import NormalizeTargets,AddMolecularFeatures


@DATASETS.register("alchemy")
class AlchemyPropertyDataset(BaseGraphDataset):
    def __init__( self,
        root="data/alchemy",
        split="train",
        normalize=False,
        stats=None,
    ):
        path = f"{root}/{split}.pt"
        self.dataset = torch.load( path, map_location="cpu" )
        assert len(self.dataset) > 0, "Empty dataset!"
        transforms = [ AddMolecularFeatures()]
        if normalize:
            if stats is None:
                stats = self.compute_stats()
            self.stats = stats
            transforms.append(NormalizeTargets(stats, keys=["y", "dipole", "polar"]))
        super().__init__( self.dataset, transform=Compose(transforms))


    def compute_stats(self):
        stats = {}
        for key in ["y", "dipole", "polar"]:
            values = []
            for item in self.dataset:
                if hasattr(item, key):
                    value = getattr(item, key)
                    if value is not None:
                        values.append(value.float())
            if values:
                x = torch.cat( values, dim=0 )
                stats[f"{key}_mean"] = x.mean()
                stats[f"{key}_std"] = x.std()
        return stats