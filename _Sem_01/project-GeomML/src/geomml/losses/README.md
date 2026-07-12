## Интерфейс лосс-функции
Каждый loss обязан иметь форму:
```python
loss(pred_tensor, target_tensor, model=None)
...
```
если аргументов больше - делать через замыкания, как в hybrid.py


## Рекомендация по многокомпонентным лоссам

Для многокомпонентных лосс-функции лучше каждую функцию делать отдельно. Например:
```mermaid
classification.py
BCELoss

symmetry.py
InvariantLoss

equivariance.py
EquivariantLoss

topology.py
PersistenceLoss

```
и потом `total.py` собирает: $L = L_cls + λ_1 * L_sym + λ_2 * L_eq + λ_3*L_tda$