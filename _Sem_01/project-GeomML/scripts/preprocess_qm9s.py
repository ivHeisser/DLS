import os,torch,sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from geomml.datasets.qm9s import *

def atom_features(z):
    atoms=[1,6,7,8,9]
    x=torch.zeros(z.size(0),len(atoms))
    for i,a in enumerate(atoms):
        x[:,i]=(z==a)
    return x.float()


def prepare(src,out):
    dataset=torch.load(src,map_location="cpu")
    result=[]
    for old in dataset:
        data=QM9SData(**old.to_dict())
        data.x=atom_features(data.z)
        data.dipole=data.dipole.reshape(1,3).float()
        data.polar=data.polar.reshape(1,9).float()
        result.append(data)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,out)
    print("saved:",out,len(result))

if __name__=="__main__":
    if len(sys.argv)<3:
        raise ValueError("Usage: python preprocess_qm9s.py <input_qm9s.pt> <output.pt>")
    prepare(sys.argv[1],sys.argv[2])