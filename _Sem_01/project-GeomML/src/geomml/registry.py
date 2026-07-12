class Registry:
    def __init__(self, name: str):
        self.name = name
        self._items = {}

    def register(self, names):

        # поддержка <str> и <list>
        if isinstance(names, str):
            names = [names]

        def decorator(obj):
            for n in names:
                ''' Запрещает замену существующей регистрации
                if n in self._items:
                    raise KeyError(
                        f"{self.name}: '{n}' already registered."
                    )
                '''
                self._items[n] = obj
            return obj
        return decorator

    def build(self, name: str, **kwargs):
        if name not in self._items:
            raise ValueError(
                f"Unknown {self.name}: {name}\n"
                f"Available: {list(self._items.keys())}"
            )
        return self._items[name](**kwargs)

    def get(self, name):
        return self._items[name]

    def names(self):
        return list(self._items.keys())

    def __contains__(self, name):
        return name in self._items

    def __len__(self):
        return len(self._items)
    

DATASETS = Registry("dataset")
MODELS = Registry("model")
LOSSES = Registry("loss")
OPTIMIZERS = Registry("optimizer")
SCHEDULERS = Registry("scheduler")
TRAINERS = Registry("trainer")
LAYERS = Registry("layer")
SYMMETRIES = Registry("symmetry")
