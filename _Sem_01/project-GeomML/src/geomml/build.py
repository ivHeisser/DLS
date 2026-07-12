from geomml.registry import DATASETS

def build(name, **kwargs):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")

    return DATASETS[name](**kwargs)

'''
# geomml/datasets/__init__.py

from .qm9 import *
from .alchemy import *
from .tda import *

или ещё лучше:

# geomml/datasets/__init__.py
from importlib import import_module

def register_all():
    import_module("geomml.datasets.qm9")
    import_module("geomml.datasets.alchemy")
    import_module("geomml.datasets.tda")

register_all()
'''