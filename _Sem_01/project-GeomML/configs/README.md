# Hydra configs


`configs/`

Все параметры экспериментов. Например:
```mermaid
mindmap

model:
    latent_dim: 128

tda:
    enabled: true
    homology_dim: 1

symmetry:
    handcrafted: true
    atlas: true

loss:
    lambda_sym: 0.2
    lambda_tda: 0.05
```

Таким образом переключение между экспериментами происходит без изменения кода.