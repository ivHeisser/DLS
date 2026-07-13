## Добавление новой модели

Добавление новой модели происходит в директорию `models`. Например:
```
models/
    egnn.py
```
Тогда внутри `egnn.py` нужно произвести импрорт python-модуля:
```python
from geomml.registry import MODELS
```
и добавить декоратор:
```python
@MODELS.register("egnn")
class EGNN(nn.Module):

    ...
```
чтобы зарегистрировать модель во фреймворке и далее в общем коде:
```python
build_model('egnn')
```

## Cоблюдение батч-интерфейса (контракта)
Dataset output:
```python
data = {
    "z": atomic_number OR fallback,
    "pos": zero if missing,
    "edge_index": shared,
    "batch": batch,
    "task_id": dataset_id,
    "y": target,
    "mask": valid_targets
    "tda": ...,
}
```
Model becomes dataset-agnostic
```python
z = batch.z
pos = batch.pos
task = batch.task_id
```

Encoder, 
Classifier, 
Fusion, 
SymmetryNet.
