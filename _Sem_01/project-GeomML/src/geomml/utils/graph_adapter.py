import torch
from torch_geometric.data import Data
from geomml.utils.pipeline import *


#  Унифицированный Adapter 
#  превращает ANY dataset → QM9-style Data(z, pos, y, task_id).
class GraphDatasetAdapter:
    """
    Unified adapter:
    - QM9 (already graph)
    - OGB (SMILES -> RDKit graph)
    - Alchemy (dict-based)
    """
    def __init__(self, dataset, adapter_fn, target_index=0, task_id=0):
        self.dataset = dataset
        self.adapter_fn = adapter_fn
        self.target_index = target_index
        self.task_id = task_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        graph = self.adapter_fn(item, idx)   

        data = Data(
            z=graph["z"],
            pos=graph["pos"],
            edge_index=graph.get("edge_index", None),
        )

        if graph.get("y", None) is not None:
            y = graph["y"]
            if y.ndim == 2:
                y = y[:, self.target_index]
            data.y = y.reshape(-1, 1)

        data.task_id = torch.tensor([self.task_id], dtype=torch.long)

        return data
    
    from torch_geometric.data import Data


class AutoGraphAdapter:
    '''
    автоматическое определение типа:
    Dataset         --- detected    ---    pipeline
    * QM9           --- 3D          ---	ThreeDPipeline
    * OGB molhiv    --- graph       --- GraphPipeline
    * SMILES dataset--- smiles      ---	RDKitPipeline
    '''
    def __init__(self, dataset):
        self.dataset = dataset

        self.dataset_type = self.infer_dataset_type()


        if self.dataset_type == "smiles":
            self.pipeline = PIPELINES.get("smiles")(dataset.smiles)
        else:
            self.pipeline = PIPELINES.get(self.dataset_type)()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        graph = self.pipeline(self.dataset, item, idx)

        data = Data(
            z=graph["z"],
            pos=graph["pos"],
            edge_index=graph.get("edge_index", None),
        )

        if graph.get("y", None) is not None:
            y = graph["y"]
            if y.ndim == 2:
                y = y[:, 0]
            data.y = y.reshape(-1, 1)

        return data
        
    def infer_dataset_type(self):
        """
        Returns: 'graph' | 'smiles' | '3d'
        """

        sample = self.dataset[0]

        # --- CASE 1: already 3D geometry ---
        if hasattr(sample, "pos") and sample.pos is not None:
            if sample.pos.ndim == 2 and sample.pos.shape[1] == 3:
                return "3d"

        # --- CASE 2: SMILES available ---
        if hasattr(self.dataset, "smiles"):
            return "smiles"

        # --- CASE 3: default graph ---
        return "graph"
