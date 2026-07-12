import torch
import torch.nn.functional as F
from geomml.registry import LOSSES


@LOSSES.register("kendall_loss")
def kendall_loss(outputs,batch,model,clamp=(-10,10)):
    pred = outputs["y"].view(-1)
    target = batch.y.view(-1)
    mask = getattr(batch, "mask_y", torch.ones_like(target)).bool()
    if mask.shape != target.shape:
        if mask.ndim == target.ndim + 1 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        else:
            raise ValueError(
                f"Unexpected mask shape: mask={mask.shape}, target={target.shape}"
            )

    task = batch.task_id.view(-1).long()
    loss = 0.0
    n_tasks = 0
    for tid in task.unique():
        idx = (task == tid) & mask
        if idx.sum()==0: continue
        mse = (pred[idx]-target[idx]).pow(2).mean()
        s = model.log_vars[tid].clamp(*clamp)
        loss += torch.exp(-s)*mse+s
        n_tasks += 1
    return loss / max(n_tasks,1)

@LOSSES.register("kendall_loss_v1")
def kendall_loss_v1(outputs,batch,model,log_var_clamp=(-10.,10.)):
    y=outputs["y"].view(-1)
    t=batch.y.view(-1)
    mask=getattr(batch,"mask_y",torch.ones_like(t)).view(-1).float()
    task=batch.task_id.view(-1).long().clamp(0,model.log_vars.numel()-1)
    log_var=torch.clamp(model.log_vars[task],*log_var_clamp)
    loss=(torch.exp(-log_var)*(y-t).pow(2)+log_var)*mask
    return loss.sum()/mask.sum().clamp(min=1.)

#===================================================================================================
#===================================================================================================
def masked_mse(pred, target, mask):
    """MSE с учетом маски."""
    diff = (pred - target).pow(2) * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom


def masked_mae(pred, target, mask):
    """MAE с учетом маски."""
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom



def kendall_loss_masked(outputs, batch, model,
                 log_var_clamp=(-10.0, 10.0),
                 eps=1e-8):
    """
    #### Источник: Kendall et al., 'Multi-Task Learning Using Uncertainty to Weigh Losses'
    1st place
    Model MUST have: self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    * написан для архитектуры с несколькими выходными головами.
    """
    log_vars = model.log_vars  # shape: [num_tasks]
    # стабилизация log_vars
    log_vars = torch.clamp(log_vars, log_var_clamp[0], log_var_clamp[1])
    loss = 0.0
    task_idx = 0

    # -------------------
    # Y task (MSE)
    # -------------------
    if hasattr(batch, "y"):
        mask = getattr(batch, "mask_y", torch.ones_like(batch.y))
        l_y = masked_mse(outputs["y"], batch.y, mask)
        precision = torch.exp(-log_vars[task_idx])
        loss += precision * l_y + log_vars[task_idx]
        task_idx += 1

    # -------------------
    # Dipole task (MAE)
    # -------------------
    if hasattr(batch, "dipole"):
        mask = getattr(batch, "mask_dipole", torch.ones_like(batch.dipole))
        l_d = masked_mae(outputs["dipole"], batch.dipole, mask)
        precision = torch.exp(-log_vars[task_idx])
        loss_d = precision * l_d + log_vars[task_idx]
        loss = loss + loss_d
        task_idx += 1

    # -------------------
    # Polar task (MSE)
    # -------------------
    if hasattr(batch, "polar"):
        mask = getattr(batch, "mask_polar", torch.ones_like(batch.polar))
        l_p = masked_mse(outputs["polar"], batch.polar, mask)
        precision = torch.exp(-log_vars[task_idx])
        loss_p = precision * l_p + log_vars[task_idx]
        loss = loss + loss_p
        task_idx += 1

    return loss




def kendall_prior_loss(outputs, batch, model,
    lam=0.1,
    reg_weight=0.01,
    log_var_clamp=(-10.0, 10.0),
):
    """
    Kendall uncertainty weighting + multitask prior.

    L = Σ(exp(-s_i) * L_i + s_i) + λ * L_prior + reg

    Model MUST have: self.log_vars = nn.Parameter(torch.zeros(num_tasks))
    * написан для архитектуры с несколькими выходными головами.
    """

    # Ограничиваем log_vars только при вычислении loss
    log_vars = model.log_vars.clamp(*log_var_clamp)

    loss_kendall = 0.0
    task_idx = 0

    # -----------------------
    # Y (MSE)
    # -----------------------
    if hasattr(batch, "y"):
        mask = getattr(batch, "mask_y", torch.ones_like(batch.y))

        l_y = masked_mse(outputs["y"], batch.y, mask)

        precision = torch.exp(-log_vars[task_idx])
        loss_kendall += precision * l_y + log_vars[task_idx]
        task_idx += 1

    # -----------------------
    # Dipole (MAE)
    # -----------------------
    if hasattr(batch, "dipole"):
        mask = getattr(batch, "mask_dipole", torch.ones_like(batch.dipole))

        l_d = masked_mae(outputs["dipole"], batch.dipole, mask)

        precision = torch.exp(-log_vars[task_idx])
        loss_kendall += precision * l_d + log_vars[task_idx]
        task_idx += 1

    # -----------------------
    # Polar (MSE)
    # -----------------------
    if hasattr(batch, "polar"):
        mask = getattr(batch, "mask_polar", torch.ones_like(batch.polar))

        l_p = masked_mse(outputs["polar"], batch.polar, mask)

        precision = torch.exp(-log_vars[task_idx])
        loss_kendall += precision * l_p + log_vars[task_idx]
        task_idx += 1

    # -----------------------
    # Prior (multitask stabilizer)
    # -----------------------
    loss_prior = model.loss_fn(outputs, batch)

    # L2-регуляризация исходных параметров log_vars
    reg = reg_weight * model.log_vars.pow(2).sum()

    return loss_kendall + lam * loss_prior + reg
#===================================================================================================
#===================================================================================================
def kendall_loss_v2(outputs, batch, model):
    """
    #### Источник: Kendall et al., 'Multi-Task Learning Using Uncertainty to Weigh Losses'
    2nd place
    Model MUST have: self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    * написан для архитектуры с несколькими выходными головами.
    """
    loss = 0.0
    task_list = list(model.heads.keys())
    for i, task in enumerate(task_list):
        if not hasattr(batch, task):
            continue
        pred = outputs[task].view(-1)
        target = getattr(batch, task).view(-1)
        mask = ~torch.isnan(target)
        if mask.sum() == 0:
            continue
        '''
        использование SmoothL1 не совсем соответствует статье Kendall:
        там предполагается взвешивание MSE для гауссовского шума или CE для классификации. 
        '''
        task_loss = F.smooth_l1_loss(pred[mask], target[mask], reduction="mean")
        precision = torch.exp(-model.log_vars[i])
        loss = loss + precision * task_loss + model.log_vars[i]

    return loss

#===================================================================================================
#===================================================================================================
def kendall_loss_v3(outputs, batch, model=None):
    '''
    Источник: Kendall et al., Multi-Task Learning Using Uncertainty to Weigh Losses
    3rd place
    Model MUST have: self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    * написан для архитектуры с несколькими выходными головами.
    '''
    loss = 0.0
# unpack uncertainty
    log_vars = model.log_vars # torch.nn.Parameter(torch.zeros(3))  # y, dipole, polar
# Y
    if hasattr(batch, "y"):
        mask = getattr(batch, "mask_y", torch.ones_like(batch.y))
        l_y = ((outputs["y"] - batch.y) ** 2 * mask).sum() / mask.sum()
        precision = torch.exp(-log_vars[0])
        loss += precision * l_y + log_vars[0]
# dipole (MAE)
    if hasattr(batch, "dipole"):
        mask = getattr(batch, "mask_dipole", torch.ones_like(batch.dipole))
        l_d = (torch.abs(outputs["dipole"] - batch.dipole) * mask).mean()
        precision = torch.exp(-log_vars[1])
        loss += precision * l_d + log_vars[1]
# polar
    if hasattr(batch, "polar"):
        mask = getattr(batch, "mask_polar", torch.ones_like(batch.polar))
        l_p = ((outputs["polar"] - batch.polar) ** 2 * mask).sum() / mask.sum()
        precision = torch.exp(-log_vars[2])
        loss += precision * l_p + log_vars[2]

    return loss