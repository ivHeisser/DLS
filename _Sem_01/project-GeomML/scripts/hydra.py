import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate

from geomml.models import build_model
from geomml.trainers.multi_task import multi_task
from geomml.datasets import build as build_dataset
from geomml.utils.loader import build_dataloader

import torch

@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):

    dataset = build_dataset(cfg.dataset.name, root=cfg.dataset.root)
    print(f"Dataset size: {len(dataset)}")
    print(cfg)

    # DEVICE
    device = cfg.training.device
    device = torch.device(cfg.training.device)

    # MODEL
    # model = build_model(
    #    cfg.model.name,
    #    **cfg.model
    # )
    # MODEL
    model = instantiate(cfg.model)

    # OPTIMIZER
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    # SCHEDULER
    scheduler = instantiate(cfg.scheduler, optimizer=optimizer)

    # DATA
    train_loader, valid_loader = build_dataloader(cfg.data)

    # TRAIN
    multi_task(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        criterion=cfg.loss_fn,
        scheduler=scheduler,
        device=device,
        **cfg.training,
        checkpoint_cfg=cfg.checkpoint
    )



if __name__ == "__main__":
    main()