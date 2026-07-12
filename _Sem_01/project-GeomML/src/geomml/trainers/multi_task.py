import torch, os, time, copy, hashlib
from tqdm import tqdm
from geomml.utils.device import *


class MultiTaskTrainer:
    def __init__(
        self,
        model,
        optimizer,
        criterion=None,
        device=get_device(),
        scheduler=None,
        min_delta=1e-4,
        patience=10,
        checkpoint_name=None,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = criterion
        self.scheduler = scheduler
        self.device = device
        self.min_delta = min_delta
        self.patience = patience
        self.best_val_loss = float("inf")
        self.checkpoint = checkpoint_name
        self.counter = 0
        
        # create directory for best_states if doesn't exist
        self.best_state_dir = "checkpoints"
        os.makedirs(self.best_state_dir, exist_ok=True)

        if self.checkpoint is not None:
            unique_hash = hashlib.md5(
                str(time.time_ns()).encode("utf-8")
            ).hexdigest()[:8]
            self.checkpoint = f"{self.checkpoint}-{unique_hash}.pth"
    
    # ---------------- utils ----------------
    def to_device(self, batch):
        if hasattr(batch, "to"):  # PyG Batch
            return batch.to(self.device)

        if isinstance(batch, dict):
            return {
                k: (v.to(self.device) if torch.is_tensor(v) else v)
                for k, v in batch.items()
            }
        return batch

    # ---------------- eval epoch ----------------
    @torch.no_grad()
    def eval_epoch(self, loader):
        self.model.eval()
        total_loss = 0.0
        n = 0
        for batch in loader:
            batch = self.to_device(batch)
            pred = self.model(batch)
            loss = self.loss_fn(pred, batch, self.model) if self.loss_fn is not None else self.model.loss_fn(pred, batch)
            total_loss += loss.item()
            n += 1
        return {"loss": total_loss / n}
    
    @torch.no_grad()
    def eval_epoch_old(model, loader, device, criterion, loss_fn):
        total_mse = 0.0
        total_mae = 0.0
        total_mae_updated = 0.0
        n_batches = 0
        total_count = 0
        for z,pos,mask,y in loader:
            z=z.to(device)
            pos=pos.to(device)
            mask=mask.to(device)
            y=y.to(device)
            pred=model(z,pos,mask)

            mse_loss = criterion(pred, y)
            mae_loss = loss_fn(pred, y)

            total_mse += mse_loss.item()
            total_mae += mae_loss.item()      
            total_mae_updated += torch.sum(torch.abs(pred - y)).item()
            total_count += pred.numel()
            n_batches += 1
        return {
            "mse": total_mse / n_batches,
            "mae": total_mae / n_batches,
            "mae_updated": total_mae_updated / total_count,
        }

    # ---------------- fit epoch ----------------
    def fit_epoch(self, loader):
        self.model.train()
        total = 0.0
        n = 0
        for batch in loader:
            batch = self.to_device(batch)
            pred = self.model(batch)
            loss = self.loss_fn(pred, batch, self.model) if self.loss_fn is not None else self.model.loss_fn(pred, batch)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            with torch.no_grad():
                self.model.log_vars.clamp_(-10,10)
            total += loss.item()
            n += 1
        return {"loss": total / n}

    def fit_epoch_old(model, loader, optimizer, criterion, device):
        model.to(device)
        model.train()
        total_loss = 0
        for z,pos,mask,y in tqdm(loader, disable=True):
            z = z.to(device) # batch = to_device(batch, device)
            pos = pos.to(device)
            mask = mask.to(device)
            y = y.to(device)
            pred = model(z,pos,mask) #   outputs = model(batch)
            loss = criterion(pred,y) # loss = task(outputs, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                model.log_vars.clamp_(-10,10)
            total_loss += loss.item()
        return total_loss/len(loader)
    
    # ---------------- full training ----------------
    def fit(self, train_loader, valid_loader, num_epochs=1000):
        train_losses, valid_losses = [], []
        for epoch in range(num_epochs):
            print(f"Epoch [{epoch+1:>{len(str(num_epochs))}}/{num_epochs}]", end=" | ")
            start_time = time.perf_counter()

            # -------- TRAIN --------
            train_loss = self.fit_epoch(train_loader)["loss"]
            # -------- VALID --------
            valid_loss = self.eval_epoch(valid_loader)["loss"]

            # -------- SCHEDULER --------
            if self.scheduler is not None:
                self.scheduler.step(valid_loss)

            # -------- EARLY STOPPING --------
            is_saved = False
            if self.best_val_loss - valid_loss > self.min_delta:
                self.best_val_loss = valid_loss
                self.best_state = copy.deepcopy(self.model.state_dict())
                if self.checkpoint is not None:
                    checkpoint_path = os.path.join(self.best_state_dir, self.checkpoint)
                    torch.save(self.best_state, checkpoint_path)
                    is_saved = True
                self.counter = 0
            else:
                self.counter += 1

            current_lr = self.optimizer.param_groups[0]["lr"]
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(
                f"TrainLoss: {train_loss:.6f}",
                f"ValidLoss: {valid_loss:.6f}",
                f"LR: {current_lr:.2e}",
                f"EarlyStop: {self.counter:^{len(str(self.patience))}}/{self.patience}" if self.patience >= 0 else "",
                f"EpochTime: {time.perf_counter() - start_time:.2f}s",
                f"Best Model State was Saved" if is_saved else "",
                sep=" | "
            )

            if self.counter >= self.patience:
                print("Early stopping triggered(!)")
                break

        # restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        
        print("\nlog_vars.data: ", self.model.log_vars.data)
        return self.model, {"train_loss": train_losses, "valid_loss": valid_losses}