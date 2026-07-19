import torch
from torch_geometric.transforms import Compose
from geomml.registry import DATASETS
from geomml.utils.base_graph import BaseGraphDataset
from geomml.utils.transforms import AddMolecularFeatures,NormalizeTargets

from torch_geometric.data import Data

class QM9SData(Data):
    def __cat_dim__(self,key,value,*args,**kwargs):
        if key in [ "dipole", "polar", "energy", "quadrupole", "octapole", "hyperpolar" ]:
            return None
        return super().__cat_dim__(key,value,*args,**kwargs)
    
class FilterDegeneratePolar:
    def __init__(self, det_eps=1e-8):
        self.det_eps = det_eps

    def __call__(self, dataset):
        filtered = []
        for data in dataset:
            P = data.polar.view(3, 3).float()
            if torch.det(P).abs() >= self.det_eps:
                filtered.append(data)
        return filtered


@DATASETS.register("qm9s")
class QM9SDataset(BaseGraphDataset):
    def __init__(self,root="data/qm9s_processed.pt",normalize=False,stats=None):
        self.dataset=torch.load(root,map_location="cpu")
        self.dataset = FilterDegeneratePolar(det_eps=1e-8)(self.dataset)
        assert len(self.dataset)>0,"Empty dataset!"
        transforms=[AddMolecularFeatures()]
        if normalize:
            if stats is None:
                stats=self.compute_stats()
            self.stats=stats
            transforms.append(NormalizeTargets(stats,keys=["dipole","polar"]))
        super().__init__(self.dataset,transform=Compose(transforms))
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        data.task_id = torch.tensor( 0, dtype=torch.long )
        return data

    def compute_stats(self):
        stats={}
        for key in ["dipole","polar"]:
            x=torch.cat([getattr(d,key).view(1,-1).float() for d in self.dataset],dim=0)
            stats[f"{key}_mean"]=x.mean(0)
            stats[f"{key}_std"]=x.std(0)
        return stats