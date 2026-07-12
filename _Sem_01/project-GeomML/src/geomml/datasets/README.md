## Регистрация датасета
```python
from geomml.registry import DATASETS

@DATASETS.register("qm9")
class QM9Dataset:

    ...
```

## Подготовка датасета в зависимости от типа задачи
Цель - сделать:
* no PyG dependency
* extensible (TDA already included)

Поддержка фреймовка нескольких типов задач. Например:
* scalar prediction	(QM9, ESOL)
* multi-property regression	(Alchemy)

Тогда нужно дальше сделать:
```mermaid
BaseGraphDataset
 ├── ScalarGraphDataset   (QM9, MoleculeNet)
 └── MultiTaskGraphDataset (Alchemy)
```
или ещё лучше:
* GraphSample
* GraphBatch
* PropertySpec (registry)

## Формат вывода датасета
Dataset output:
```mermaid
data = {
    "z": atomic_number OR fallback,
    "pos": zero if missing,
    "edge_index": shared,
    "batch": batch,
    "task_id": dataset_id,
    "y": target,
    "mask": valid_targets
    "tda"
    "targets"  # vector or dict
}
``` 
 Model becomes dataset-agnostic:
```mermaid
z = batch.z
pos = batch.pos
task = batch.task_id
...
```



TODO: research-grade system:
1. smart sampling: temperature-based dataset balancing
2. task conditioning: transformer with task tokens
3. graph batching (PyG-style but custom)
4. HDF5 / LMDB caching (10–50x speedup)

TODO:  “production-grade” architecture:
5. GraphBatch dataclass вместо dict
6. support for padding / batching molecules
7. automatic normalization of targets (mean/std per dataset)
8. cached preprocessing (speed x10–100)
9. unified edge construction (QM9 vs MoleculeNet differences)
10. сделать через единый type inference слой. Идея: один раз “распознается” датасет → дальше вся система работает автоматически. Для любого dataset получить:
```mermaid
dataset_type ∈ {
    "graph",
    "smiles",
    "3d"
}
```
и уже от этого выбирать pipeline:
```mermaid
graph → OGB-style
smiles → RDKit 2D→3D
3d → QM9 / GEOM-style
```