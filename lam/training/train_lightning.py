import os
import sys
import logging
from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lam.dataset.cafca_lam_dataset import CafcaLamDataset
# Ensure this import points to your LightningModule file and the correct get_logger
# If get_logger in lightning_lam_cafca.py is the one from train_lam_cafca.py, that's fine.
from lam.training.lightning_lam_cafca import LamLightningModel 

# Local get_logger definition for this script
def get_logger(name, level=logging.INFO):
    """Initializes and returns a logger."""
    logger_instance = logging.getLogger(name)
    logger_instance.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger_instance.hasHandlers():
        logger_instance.addHandler(handler)
    logger_instance.propagate = False
    return logger_instance

logger = get_logger(__name__)

def train(cfg: DictConfig):
    """Main training loop using PyTorch Lightning."""
    exp_name = cfg.experiment.name
    
    # WandB Logger
    wandb_logger = None
    if cfg.wandb.enabled:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.get("run_name", exp_name),
            log_model="all" if cfg.wandb.get("log_model_checkpoints", False) else False,
            save_dir=str(Path(cfg.experiment.output_dir) / exp_name) 
        )
        if wandb_logger.experiment is not None: 
             wandb_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))

    # Datasets and DataLoaders
    logger.info("Initializing datasets...")
    train_dataset = CafcaLamDataset(
        subject_list=list(cfg.dataset.cafca_subject_ids_train),
        num_source_frames=cfg.dataset.num_of_src_views,
        num_driving_frames=cfg.dataset.num_of_target_views,
        image_size=cfg.training.image_size,
        is_val=False
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    logger.info(f"Train dataset size: {len(train_dataset)}. Train Dataloader size: {len(train_dataloader)} batches.")

    val_dataloader = None
    if cfg.dataset.cafca_subject_ids_val and len(cfg.dataset.cafca_subject_ids_val) > 0:
        val_dataset = CafcaLamDataset(
            subject_list=list(cfg.dataset.cafca_subject_ids_val),
            num_source_frames=cfg.dataset.num_of_src_views,
            num_driving_frames=cfg.dataset.num_of_target_views,
            image_size=cfg.training.image_size,
            is_val=True
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=True
        )
        logger.info(f"Validation dataset size: {len(val_dataset)}. Val Dataloader size: {len(val_dataloader)} batches.")

    # Lightning Model
    logger.info("Initializing LightningModel...")
    lightning_model = LamLightningModel(cfg)

    # Callbacks
    callbacks = []
    if wandb_logger:
        lr_monitor = LearningRateMonitor(logging_interval='step')
        callbacks.append(lr_monitor)

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(Path(cfg.experiment.output_dir) / exp_name / "checkpoints"),
        filename='{epoch:04d}-{val/total_loss:.4f}',
        save_top_k=cfg.training.get("save_top_k_checkpoints", 3),
        monitor="val/total_loss", 
        mode="min",
        save_last=True, 
        every_n_epochs=cfg.training.save_every_n_epochs
    )
    callbacks.append(checkpoint_callback)

    # Trainer
    logger.info("Starting training...")
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.training.num_epochs,
        accelerator=cfg.training.device, 
        devices=1,
        precision=cfg.training.get("precision", "32-true"), 
        log_every_n_steps=cfg.wandb.log_every_n_steps,
        check_val_every_n_epoch=cfg.training.get("validate_every_n_epochs", 1.0),
    )

    trainer.fit(model=lightning_model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader)

    logger.info("Training finished.")

if __name__ == "__main__":    
    config_path_str = sys.argv[1] if len(sys.argv) > 1 else "configs/training/train_lam_cafca.yaml"
        
    cfg = OmegaConf.load(config_path_str)
    
    # Allow overriding config values from the command line
    cli_overrides = OmegaConf.from_cli(sys.argv[2:])
    cfg = OmegaConf.merge(cfg, cli_overrides)
    
    logger.info("Configuration loaded:")
    logger.info(OmegaConf.to_yaml(cfg))

    train(cfg)
