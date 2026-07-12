import math
import torch
import torch.nn.functional as F
from collections import defaultdict


@torch.no_grad()
def evaluate_model(model, loader, device=None):
    model.eval()

    if device is None:
        device = next(model.parameters()).device

    stats = defaultdict(lambda: {
        "sse": 0.0,      # sum squared errors
        "sae": 0.0,      # sum absolute errors
        "n": 0
    })

    for batch in loader:
        batch = batch.to(device)

        pred = model(batch)["y"].view(-1)
        target = batch.y.view(-1)
        task = batch.task_id.view(-1)

        mask = getattr(batch, "mask_y", torch.ones_like(target)).bool()
        if mask.ndim == target.ndim + 1 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        assert pred.shape == target.shape == mask.shape == task.shape, (
            f"Shape mismatch: pred={pred.shape}, "
            f"target={target.shape}, mask={mask.shape}, task={task.shape}"
        )
        pred = pred[mask]
        target = target[mask]
        task = task[mask]

        for tid in task.unique():
            idx = task == tid

            err = pred[idx] - target[idx]

            stats[int(tid)]["sse"] += err.pow(2).sum().item()
            stats[int(tid)]["sae"] += err.abs().sum().item()
            stats[int(tid)]["n"] += idx.sum().item()

    per_task = {}

    total_sse = 0
    total_sae = 0
    total_n = 0

    for tid, s in sorted(stats.items()):

        mse = s["sse"] / s["n"]
        rmse = math.sqrt(mse)
        mae = s["sae"] / s["n"]

        per_task[tid] = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "n": s["n"]
        }

        total_sse += s["sse"]
        total_sae += s["sae"]
        total_n += s["n"]

    overall_mse = total_sse / total_n
    overall_rmse = math.sqrt(overall_mse)
    overall_mae = total_sae / total_n

    return {
        "overall": {
            "mse": overall_mse,
            "rmse": overall_rmse,
            "mae": overall_mae,
        },
        "per_task": per_task,
    }


def print_metrics(model, test_loader):
    metrics = evaluate_model(model, test_loader)

    print("\nOverall:")
    for k, v in metrics["overall"].items():
        print(f"{k:5s}: {v:.6f}")

    print("\nPer task:")
    for tid, m in metrics["per_task"].items():
        print(
            f"Task {tid:2d} | "
            f"MSE={m['mse']:.6f}\t"
            f"RMSE={m['rmse']:.6f}\t"
            f"MAE={m['mae']:.6f}\t"
        )