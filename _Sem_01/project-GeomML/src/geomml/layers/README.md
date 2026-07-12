# layers
Идея папки /layers — не хранить целые модели, а хранить переиспользуемые строительные блоки, из которых собираются разные архитектуры.
У NequIP похожая идея: есть отдельные модули для сферических гармоник, радиальных базисов, свёрток и т.д., а модель лишь комбинирует их.

Например:
```
repo/
│
├── models/
│   ├── nequip.py
│   ├── egnn.py
│   ├── schnet.py
│   ├── equiformer.py
│   └── molecular_model.py
│
├── layers/
│   ├── radial_basis.py
│   ├── cutoff.py
│   ├── message_passing.py
│   ├── interaction_block.py
│   ├── equivariant_linear.py
│   ├── pooling.py
│   ├── attention.py
│   └── normalization.py
    ├── embedding.py , activation.py , scaling.py
```
Например, один и тот же RadialBasisExpansion может использоваться и в NequIP, и в SchNet, и в PaiNN.

## Что обычно лежит в /layers

Для molecular GNN это могут быть:
```
layers/
    embedding.py          # embedding атомов
    radial_basis.py       # Gaussian/Bessel RBF
    cutoff.py             # Cosine cutoff
    edge_features.py      # вычисление edge features
    interaction.py        # interaction block
    message_passing.py    # общий MessagePassing
    equivariant.py        # E(3)-equivariant операции
    attention.py          # attention block
    pooling.py            # global pooling
    mlp.py                # универсальный MLP
    residual.py           # residual block
    norm.py               # LayerNorm, GraphNorm и т.п.
```

Тогда модель выглядит очень компактно:
```
class NequIP(nn.Module):
    def __init__(self):
        self.embedding = AtomEmbedding(...)
        self.rbf = GaussianRBF(...)
        self.layers = nn.ModuleList([
            InteractionBlock(...)
            for _ in range(6)
        ])
        self.pool = GlobalPool()
        self.head = EnergyHead()
```
Все сложные детали находятся в `layers/`.

## Поддержка несколько моделей
Например:
 - EGNN
 - NequIP
 - SchNet
 - PaiNN
 - GemNet
 - Equiformer

Поскольку, примерно, 70–80% кода между ними совпадает (эмбеддинги, RBF, pooling, residual-блоки, MLP, нормализация и т.д.). Вынос этих компонентов в /layers позволяет избежать дублирования.

## Альтернативный вариант
Поскольку цель — исследовательский код для молекулярных GNN, но без излишнего усложнения, то разумный компромисс выглядит так:
```
repo/
├── models/
│   ├── egnn.py
│   ├── nequip.py
│   └── molecular.py
│
├── blocks/
│   ├── interaction.py
│   ├── mlp.py
│   ├── pooling.py
│   ├── embedding.py
│   └── residual.py
│
├── losses/
├── trainers/
├── datasets/
├── transforms/
├── utils/
└── configs/
```
Название `blocks/` часто лучше отражает смысл, чем `layers/` - это именно крупные переиспользуемые строительные блоки модели, а не только отдельные слои nn.Module. 
Такой подход остаётся компактным, но хорошо масштабируется, если позже появятся новые архитектуры.