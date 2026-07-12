from rdkit import Chem
from rdkit.Chem import AllChem
import torch


class RDKitGraphBuilder:
    """
    Converts SMILES -> graph with:
    - z (atomic numbers)
    - pos (3D conformer)
    - edges (optional if needed)
    """

    @staticmethod
    def mol_from_smiles(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        return mol

    @staticmethod
    def get_3d_conformer(mol, seed=0):
        AllChem.EmbedMolecule(mol, randomSeed=seed)
        AllChem.UFFOptimizeMolecule(mol)
        return mol

    @staticmethod
    def extract_z(mol):
        return torch.tensor(
            [atom.GetAtomicNum() for atom in mol.GetAtoms()],
            dtype=torch.long
        )

    @staticmethod
    def extract_pos(mol):
        conf = mol.GetConformer()
        pos = []
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            pos.append([p.x, p.y, p.z])
        return torch.tensor(pos, dtype=torch.float)