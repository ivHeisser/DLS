import importlib
import pkgutil
from geomml.registry import MODELS

# автоматически импортируем все модули models
for _, module_name, _ in pkgutil.iter_modules(__path__):
    try:
        # print("Import:", module_name)
        importlib.import_module(f"{__name__}.{module_name}")
    except Exception as e:
        print(module_name, e)

def build(name, **kwargs):
    return MODELS.build(name, **kwargs)