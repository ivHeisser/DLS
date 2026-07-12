from torch_geometric.data import Data
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch
from torch.utils.data import Dataset


class BaseGraphDataset(Dataset):
    """
    Unified molecular graph interface:

    ALWAYS guarantees:
    - z (atomic number)
    - pos (3D or zeros)
    - edge_index (bond between atoms)
    - task_id (multitasking)
    - y (optional)
    """

    def __init__(self, dataset, target_index=None, task_id=0, transform=None):
        self.dataset = dataset
        self.target_index = target_index
        self.task_id = task_id
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    # -------------------------
    # ATOM FEATURES (CRITICAL FIX)
    # -------------------------
    def extract_z(self, item):
        # dict dataset (Alchemy / precomputed)
        if isinstance(item, dict):
            if "z" in item:
                return torch.tensor(item["z"], dtype=torch.long)

            if "x" in item:
                return torch.tensor(item["x"]).argmax(dim=-1).long()

            raise ValueError("No z/x in dict item")
        # PyG datasets
        if hasattr(item, "z"):
            return item.z.long()
        if hasattr(item, "x"):
            return item.x.argmax(dim=-1).long()
        raise ValueError(f"No atomic features in {type(item)}")

    # -------------------------
    # POSITIONS (SAFE FALLBACK)
    # -------------------------
    def extract_pos(self, item, num_nodes):
        pos = getattr(item, "pos", None)
        if isinstance(pos, torch.Tensor):
            return pos.float()
        if pos is not None:
            try:
                return torch.tensor(pos, dtype=torch.float)
            except:
                pass
        return torch.zeros((num_nodes, 3), dtype=torch.float)

    # -------------------------
    # EDGES (IMPORTANT FIX)
    # -------------------------
    def extract_edges(self, item):
        if hasattr(item, "edge_index"):
            return item.edge_index
        if isinstance(item, dict) and "edge_index" in item:
            return torch.tensor(item["edge_index"])
        raise ValueError("No edge_index found")

    # -------------------------
    # TARGETS
    # -------------------------
    def extract_targets_(self, item):
        if isinstance(item, dict):
            out = {}
            for k in ["y", "dipole", "polar", "tda"]:
                if k in item:
                    out[k] = torch.tensor(item[k]).float().view(-1, 1)
            return out

        if hasattr(item, "y") and item.y is not None:
            y = item.y.float()
            if self.target_index is not None:
                if y.ndim == 2:
                    y = y[:, self.target_index]
                else:
                    y = y[self.target_index]
            return {"y": y.view(-1, 1)}
        return {}

    def extract_targets(self, item):
        out = {}
# dict dataset (например Alchemy .pt)
        if isinstance(item, dict):
            for k in ["y", "dipole", "polar", "tda"]:
                if k in item:
                    out[k] = torch.tensor(item[k]).float().view(-1, 1)
            return out

# PyG Data dataset (например QM9 / Alchemy после преобразования)
        for k in ["y", "dipole", "polar", "tda"]:
            if hasattr(item, k):
                value = getattr(item, k)
                if value is None:
                    continue
                value = value.float()
# только для y применяем выбор target_index
                if k == "y" and self.target_index is not None:
                    if value.ndim == 2:
                        value = value[:, self.target_index]
                    else:
                        value = value[self.target_index]
                out[k] = value.view(-1, 1)
        return out
    # -------------------------
    # MAIN
    # -------------------------
    def __getitem__(self, idx):
        item = self.dataset[idx]
# atomic size first (IMPORTANT)
        num_nodes = len(self.extract_z(item))
        data = Data(
            z=self.extract_z(item),
            pos=self.extract_pos(item, num_nodes),
            edge_index=self.extract_edges(item),
        )
# targets
        for k, v in self.extract_targets(item).items():
            setattr(data, k, v)
# task metadata
        data.task_id = torch.tensor(self.task_id, dtype=torch.long)
# tranforms
        if self.transform is not None: 
            data = self.transform(data)
        return data
    
    # -------------------------
    # TRANSFORM DATASET
    # -------------------------
    def set_transform(self, transform):
        self.transform = transform