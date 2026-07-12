from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class RegressionModel(nn.Module):
    """
    Универсальная класс-обёртка.

    Если tda_cache=None:
        используется обычный EGNN.

    Если передан TDACache:
        используется EGNN+TDA.
    """

    def __init__(
        self,
        backbone: nn.Module,
        device: torch.device,
        tda_cache=None,
    ):
        super().__init__()

        self.backbone = backbone
        self.device = device

        self.tda_cache = tda_cache
        self.use_tda = tda_cache is not None

    def _load_tda(self, idxs):

        vecs = []

        for idx in idxs.tolist():

            path = self.tda_cache.path_for_idx(int(idx))

            if not path.exists():
                raise FileNotFoundError(path)

            vecs.append(np.load(path))

        x = np.stack(vecs)

        return torch.from_numpy(x).float().to(self.device)

    def forward(self, batch):

        z = batch.z.to(self.device)
        pos = batch.pos.to(self.device)
        mask = batch.mask.to(self.device)

        if self.use_tda:

            tda = self._load_tda(batch.idx)

            return self.backbone(
                z,
                pos,
                mask,
                tda,
            )

        return self.backbone(
            z,
            pos,
            mask,
        )