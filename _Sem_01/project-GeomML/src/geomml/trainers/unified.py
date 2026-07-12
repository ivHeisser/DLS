import torch, os, time, copy
from tqdm import tqdm


class UnifiedTrainer:
    '''
    универсальный (multi-task) тренировочный цикл
    '''
    def __init__(self, model, optimizer, loss_fn_dict, device="cuda"):
        """
        loss_fn_dict:
            {
                "y": mse_loss,
                "dipole": mse_loss,
                "polar": mse_loss,
            }
        """
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn_dict = loss_fn_dict
        self.device = device

    # ---------------- train step ----------------

    def train_step(self, batch):
        batch = self._to_device(batch)
        pred = self.model(batch)
        loss = 0.0
        logs = {}
        for key, loss_fn in self.loss_fn_dict.items():
            if key not in batch or key not in pred:
                continue
            y_true = batch[key]
            y_pred = pred[key]
            l = loss_fn(y_pred, y_true)
            loss = loss + l
            logs[key] = l.item()
        logs["loss"] = loss.item()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return logs

    # ---------------- epoch loop ----------------
    def fit_epoch(self, loader):
        self.model.train()
        pbar = tqdm(loader)
        total_logs = {}
        for batch in pbar:
            logs = self.train_step(batch)
            for k, v in logs.items():
                total_logs[k] = total_logs.get(k, 0) + v
            pbar.set_postfix(logs)
        for k in total_logs:
            total_logs[k] /= len(loader)
        return total_logs

    # ---------------- utils ----------------
    def _to_device(self, batch):
        def move(x):
            if torch.is_tensor(x):
                return x.to(self.device)
            return x
        return {k: move(v) for k, v in batch.items()}
    
