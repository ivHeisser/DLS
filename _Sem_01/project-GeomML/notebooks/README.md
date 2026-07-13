# Создание нового ноутбука 
Скопировать в первую ячейку ноутбука код:

```python
from pathlib import Path
import os

root = Path(os.getcwd()).resolve().parents[1]  # вверх на 2 уровня
!python {str(root / "scripts" / "bootstrap.py")}

%load_ext autoreload
%autoreload 2

from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

config_name = "config_qm9"

with initialize_config_dir(config_dir=str(root / "configs"), version_base="1.3"):
    cfg = compose(config_name=config_name)
```

Он необходим, чтобы при перезапусках обновлялся список submodules и регистрировались новые submodules.

# Импортирование в GeomML

импортировать можно каждый submodule по отдельности:
```
from geomml.models.dimenet_lite import MolecularDimeNetLite
```
либо в целях удобства всего один раз проимпортировать общий `builder`:
```
from geomml.models import build as build_model
```
а потом уже создавать множество компонент этого submodule:
```
model_1 = build_model("dimenet_lite")
model_2 = build_model("molecular_v1")
...
```
