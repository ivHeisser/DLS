import torch
import torch.nn as nn

class BaseEncoder(nn.Module):
    @property
    def out_dim(self):
        raise NotImplementedError