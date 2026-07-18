import torch.nn.functional as F
from geomml.registry import LOSSES
import torch


@LOSSES.register("mae_mse")
def mae_mse(output_key="y", target_key="y", mae_w=0.8, mse_w=0.2): # python-closure
    def loss_fn(outputs, batch, model=None):
        pred = outputs[output_key]
        target = getattr(batch, target_key)

        return (
            mae_w * F.l1_loss(pred, target)
            + mse_w * F.mse_loss(pred, target)
        )

    return loss_fn

@LOSSES.register("multitask")
def multitask_loss(
    y_weights=(0.8, 0.2),
    dipole_weights=(1.0, 0.0),
    polar_weights=(0.5, 0.5),
):
    loss_y = mae_mse("y", "y", *y_weights)
    loss_d = mae_mse("dipole", "dipole", *dipole_weights)
    loss_p = mae_mse("polar", "polar", *polar_weights)

    def loss_fn(outputs, batch, model=None):
        loss = 0.0

        if hasattr(batch, "y"):
            loss = loss + loss_y(outputs, batch)

        if hasattr(batch, "dipole"):
            loss = loss + loss_d(outputs, batch)

        if hasattr(batch, "polar"):
            loss = loss + loss_p(outputs, batch)

        return loss

    return loss_fn


def mae(pred, target):
    return torch.mean(torch.abs(pred - target))


def mse(pred, target):
    return torch.mean((pred - target) ** 2)