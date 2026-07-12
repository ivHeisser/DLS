# scripts/preprocess_alchemy.py
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data


'''
target columns обычно такие:

gap
HOMO
LUMO
alpha
R2
U0
ZPVE
U
'''
from pathlib import Path
import torch
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from torch_geometric.data import Data


# ======================
# CONFIG
# ======================

ROOT = Path("data/alchemy")
CSV_PATH = ROOT / "final_version.csv"
OUT_PATH = ROOT / "train.pt"


# ======================
# LOAD CSV
# ======================

targets = pd.read_csv(CSV_PATH)

# нормализуем имена колонок (ВАЖНО!)
targets.columns = [
    c.split("\n")[0].strip()
    for c in targets.columns
]

print("[INFO] Columns:", targets.columns.tolist())
print("[INFO] CSV size:", len(targets))


# ======================
# FIND SDF FILES
# ======================

sdf_dirs = sorted([p for p in ROOT.iterdir() if p.is_dir() and p.name.startswith("atom_")])

sdf_files = []
for d in sdf_dirs:
    sdf_files += sorted(d.glob("*.sdf"))

sdf_files = sorted(sdf_files)

print("[INFO] SDF files:", len(sdf_files))


# safety check
assert len(sdf_files) == len(targets), (
    f"Mismatch: {len(sdf_files)} sdf vs {len(targets)} csv"
)


# ======================
# MOLECULE PARSER
# ======================

def load_mol(path):
    suppl = Chem.SDMolSupplier(str(path), removeHs=False)
    if not suppl or len(suppl) == 0:
        return None
    return suppl[0]


def extract_graph(mol):
    z = []
    pos = []

    conf = mol.GetConformer()

    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        z.append(atom.GetAtomicNum())

        p = conf.GetAtomPosition(idx)
        pos.append([p.x, p.y, p.z])

    return z, pos


# ======================
# TARGET CONFIG (ВЫБОР)
# ======================

# ВАРИАНТ 1: gap + alpha (самый стабильный)
TARGETS = ["gap", "alpha"]

# ВАРИАНТ 2: HOMO + LUMO
# TARGETS = ["HOMO", "LUMO"]


# ======================
# PROCESSING
# ======================

data_list = []

for i, row in tqdm(targets.iterrows(), total=len(targets)):

    sdf_path = sdf_files[i]

    mol = load_mol(sdf_path)
    if mol is None:
        continue

    try:
        z, pos = extract_graph(mol)
    except Exception:
        continue

    y = torch.tensor([row["gap"]], dtype=torch.float32)

    data_list.append(
        Data(
            z=torch.tensor(z, dtype=torch.long),
            pos=torch.tensor(pos, dtype=torch.float),
            y=y,
        )
    )


# ======================
# FINAL CHECK
# ======================

print("[INFO] Processed:", len(data_list))
assert len(data_list) > 0, "Empty dataset!"

# ======================
# SAVE
# ======================

torch.save(data_list, OUT_PATH)

print(f"[DONE] Saved to {OUT_PATH}")