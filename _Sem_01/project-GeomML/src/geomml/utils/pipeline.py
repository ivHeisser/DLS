import torch
from geomml.utils.rdkit_builder import RDKitGraphBuilder

class PipelineRegistry:
    def __init__(self):
        self._map = {}

    def register(self, key):
        def wrapper(fn):
            self._map[key] = fn
            return fn
        return wrapper

    def get(self, key):
        return self._map[key]


PIPELINES = PipelineRegistry()

@PIPELINES.register("graph")
class GraphPipeline:
    '''
    1. Graph-only (OGB)
    '''
    def __init__(self):
        pass

    def __call__(self, dataset, item, idx):
        return {
            "z": item.x.argmax(dim=-1) if item.x is not None else None,
            "pos": torch.zeros((item.num_nodes, 3)),
            "edge_index": item.edge_index,
            "y": item.y.float() if item.y is not None else None
        }
    

@PIPELINES.register("smiles")
class SmilesPipeline:
    '''
    2. SMILES → RDKit → 3D
    '''
    def __init__(self, smiles_list):
        self.smiles = smiles_list
        self.builder = RDKitGraphBuilder()

    def __call__(self, dataset, item, idx):
        smiles = self.smiles[idx]

        mol = self.builder.mol_from_smiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        mol = self.builder.get_3d_conformer(mol)

        return {
            "z": self.builder.extract_z(mol),
            "pos": self.builder.extract_pos(mol),
            "edge_index": None,
            "y": item.y.float() if item.y is not None else None
        }
    

@PIPELINES.register("3d")
class ThreeDPipeline:
    '''
    3. Pure 3D (QM9, GEOM)
    '''
    def __call__(self, dataset, item, idx):
        return {
            "z": item.z,
            "pos": item.pos,
            "edge_index": getattr(item, "edge_index", None),
            "y": item.y.float() if hasattr(item, "y") else None
        }