## Single Responsibility Principle
* тренировочный цикл отвечает только за обучение
* обёртка — за подготовку входов и выбор способа вызова модели, 
* сами модели — только за вычисление предсказаний. 
Такое разделение соответствует принципу единственной ответственности. 

Это делает код проще расширять, например, если позже появятся новые варианты модели (EGNN+RDKit, EGNN+3D descriptors, EGNN+TDA+RDKit и т.д.), не меняя код обучения.

## Cоблюдение интерфейса (контракта) между dataset и trainer

Зафиксировать Контракт/Интерфейс между данными датасета и выходами модели. Например:
```mermaid
mindmap
Dataset → dict
{
    "z": ...,
    "pos": ...,
    "y": ... or None,
    "dipole": ...,
    "polar": ...,
}

Model → dict output
{
    "y": ... or None,
    "dipole": ...,
    "polar": ...,
}
```

Преимущества подхода:
* Zero dataset-specific code in training loop - trainer вообще не знает, что за датасет
* Масштабируемость - можно добавлять, например:
    - forces (MD17)
    - charges
    - spectra
    - 3D equivariant heads

без изменения trainer’а


## TODO: 
train_epoch вообще не зависит от задачи и не знает, что находится в батче — молекулы, изображения или текст. Он лишь передает батч модели и функции потерь. 

Подумать над следующими вариантами:

#### 1. Вариант 1. Батч — словарь (самый простой). Датасет возвращает словарь
```mermaid
{
    "z": ...,
    "pos": ...,
    "mask": ...,
    "y": ...
}
```
или
```mermaid
{
    "z": ...,
    "pos": ...,
    "tda": ...,
    "dipole": ...,
    "polar": ...
}
```

Тогда модель тоже принимает словарь. Например, для GAP:
```python
class EGNNGapRegressor(nn.Module):

    ...

    def forward(self, batch):

        h = self.encoder(
            batch["z"],
            batch["pos"],
            batch["mask"],
        )

        return {
            "gap": self.head(h).squeeze(-1)
        }
```
Для диполя/поляризуемости:
```python
class MolecularModel(nn.Module):

    ...

    def forward(self, batch):

        dipole, polar = self.network(
            batch["z"],
            batch["pos"],
            batch["tda"],
        )

        return {
            "dipole": dipole,
            "polar": polar,
        }
```
И loss тоже становится независимым.

Для GAP:
```python
criterion = nn.MSELoss()

def loss_fn(outputs, batch):
    return criterion(outputs["gap"], batch["y"])
```

Для двух выходов:
```python
criterion = nn.MSELoss()

def loss_fn(outputs, batch):

    loss_d = criterion(
        outputs["dipole"],
        batch["dipole"],
    )

    loss_p = criterion(
        outputs["polar"],
        batch["polar"],
    )

    return loss_d + loss_p
```
train_epoch вообще менять не придется.

#### 2. Вариант 2. Использовать классы (более масштабируемый)

Иногда делают объект задачи.
```python
class GapTask:

    def loss(self, outputs, batch):
        return F.mse_loss(outputs["gap"], batch["y"])

    def metrics(self, outputs, batch):
        ...
```
и
```python
class DipoleTask:

    def loss(self, outputs, batch):
        ...
```
Тогда
```python
loss = task.loss(outputs, batch)
```
а
```python
metrics = task.metrics(outputs, batch)
```
Это особенно удобно, если проектов становится много.

#### 3. Вариант 3. Использовать **batch

Еще более "питоновский" способ.

Модель
```python
def forward(self, z, pos, mask):
```
а батч —
```python
batch = {
    "z": ...,
    "pos": ...,
    "mask": ...
}
```
Тогда
```python
pred = model(
    z=batch["z"],
    pos=batch["pos"],
    mask=batch["mask"],
)
```
или проще
```python
pred = model(**batch)
```
Но тогда в словаре должны быть только аргументы модели. Если там есть `y`, `dipole`, `polar`, то возникнет ошибка. Поэтому обычно делают:
```python
inputs = {
    "z": batch["z"],
    "pos": batch["pos"],
    "mask": batch["mask"],
}

pred = model(**inputs)
```
или модель сама принимает словарь целиком.