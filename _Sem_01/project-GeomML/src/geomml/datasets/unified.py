import torch
from torch_geometric.data import Data
from geomml.utils.base_graph import BaseGraphDataset
from geomml.utils.transforms import *
from geomml.registry import DATASETS

class UnifiedDataset(torch.utils.data.Dataset):
    BASE_FIELDS = [ "z", "pos", "edge_index", "edge_attr", "aromatic", "charge", ]
    IGNORE_FIELDS = ["task_id",]

    def __init__(self, datasets, 
                 normalize=False, 
                 target_fields=None,
    ):
        self.datasets = datasets
        self.lengths = [len(d) for d in datasets]
        self.cumsum = [0]
        for l in self.lengths:
            self.cumsum.append(self.cumsum[-1] + l)
# автоматически ищем target поля
        if target_fields is None:
            target_fields = set()
            for ds in datasets:
                sample = ds[0]
                for key in sample.keys():
                    if (
                        key not in self.BASE_FIELDS
                        and key not in self.IGNORE_FIELDS
                    ):
                        target_fields.add(key)
            target_fields = sorted(list(target_fields))
        self.target_fields = target_fields
# статистики
        if normalize:
            self.stats = { i: ds.compute_stats() for i, ds in enumerate(datasets) }
            self.transform = NormalizeByTask( self.stats, keys=self.target_fields )
        else:
            self.transform = None

    def __len__(self):
        return self.cumsum[-1]

    def _resolve(self, idx):
        for task_id in range(len(self.datasets)):
            if idx < self.cumsum[task_id + 1]:
                return (
                    self.datasets[task_id],
                    idx - self.cumsum[task_id],
                    task_id
                )
        raise IndexError(idx)

    def __getitem__(self, idx):
        dataset, local_idx, task_id = self._resolve(idx)
        src = dataset[local_idx]
        data = Data()
# ------------------
# graph information
# ------------------
        for key in self.BASE_FIELDS:
            if hasattr(src, key):
                data[key] = getattr(src, key)
            else:
                if key == "edge_attr":
                    data.edge_attr = torch.zeros(
                        (src.edge_index.size(1), 1),
                        dtype=torch.float
                    )
                elif key == "aromatic":
                    data.aromatic = torch.zeros(
                        src.z.size(0),
                        dtype=torch.long
                    )
                elif key == "charge":
                    data.charge = torch.zeros(
                        src.z.size(0),
                        dtype=torch.float
                    )
# ------------------
# multitask targets
# ------------------
        for key in self.target_fields:
            if hasattr(src, key):
                value = getattr(src, key)
                data[key] = value
                data[f"mask_{key}"] = torch.ones(value.shape, dtype=torch.bool)
            else:
                # неизвестный размер пока
                data[key] = torch.zeros( 1,1 )
                data[f"mask_{key}"] = torch.zeros( 1, 1, dtype=torch.bool)
        data.task_id = torch.tensor( task_id, dtype=torch.long )
        if self.transform:
            data = self.transform(data)
        return data


@DATASETS.register("unified")
def unified_builder(datasets, normalize=False):
    return UnifiedDataset(datasets, normalize)

