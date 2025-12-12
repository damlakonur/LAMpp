import os
import sys
import logging
from pathlib import Path
import random
import numpy as np

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import PyTorchProfiler
from torch.profiler import schedule, tensorboard_trace_handler
_KINETO_AVAILABLE = torch.profiler.kineto_available()
import torch.utils.benchmark as benchmark


if _KINETO_AVAILABLE:
    def _delete_profilers_fixed(self):
        # Only attempt to fetch events if profiler is present and alive
        if self.profiler is not None:
            if not self._emit_nvtx:
                # Grab events BEFORE __exit__ destroys the profiler
                try:
                    if hasattr(self.profiler, "events"):
                        self.function_events = self.profiler.events()
                    elif hasattr(self.profiler, "function_events"):
                        self.function_events = self.profiler.function_events
                except AssertionError:
                    # Profiler object was already destroyed or not initialized: just skip
                    self.function_events = None
            self.profiler.__exit__(None, None, None)
            self.profiler = None
        if self._schedule is not None:
            self._schedule.reset()
        if self._parent_profiler is not None:
            self._parent_profiler.__exit__(None, None, None)
            self._parent_profiler = None
        if self._register is not None:
            self._register.__exit__(None, None, None)
            self._register = None
    PyTorchProfiler._delete_profilers = _delete_profilers_fixed

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lam.dataset.cafca_lam_de_dataset_static import CafcaLamDataset
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
    SEED = 12345
    pl.seed_everything(SEED, workers=True)

    def seed_worker(worker_id):
        worker_seed = SEED + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    torch_gen = torch.Generator().manual_seed(SEED)
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
        generator=torch_gen,
        worker_init_fn=seed_worker,
        num_workers=cfg.training.num_workers,
        pin_memory=False,
        persistent_workers=True,
        # prefetch_factor=3
    )
    logger.info(f"Train dataset size: {len(train_dataset)}. Train Dataloader size: {len(train_dataloader)} batches.")

    val_dataloader = None
    if cfg.dataset.cafca_subject_ids_val and len(cfg.dataset.cafca_subject_ids_val) > 0:
        val_dataset = CafcaLamDataset(
            subject_list=list(cfg.dataset.cafca_subject_ids_val),
            num_source_frames=cfg.dataset.num_of_src_views,
            num_driving_frames=cfg.dataset.num_of_target_views,
            image_size=cfg.training.image_size,
            is_val=True,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            generator=torch_gen,
            worker_init_fn=seed_worker,
            num_workers=2,
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
    if cfg.profiler.get("is_enabled", True):
        trace_dir = Path("lightning_logs") / "demo_npz"
        profiler = PyTorchProfiler(
            schedule=schedule(wait=1, warmup=1, active=1, repeat=0),
            on_trace_ready=tensorboard_trace_handler(trace_dir),
            metric="self_cuda_time_total",
        )
    # Trainer
    logger.info("Starting training...")
    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=callbacks,
        max_epochs=cfg.training.get("num_epochs", 100),
        accelerator=cfg.training.device, 
        precision=cfg.training.get("precision", "16-mixed"), 
        check_val_every_n_epoch=cfg.training.get("validate_every_n_epochs", 1.0),
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
        # profiler= profiler if cfg.profiler.get("is_enabled") else None,
    )

    trainer.fit(model=lightning_model, 
                train_dataloaders=train_dataloader, 
                val_dataloaders=val_dataloader)

    logger.info("Training finished.")

if __name__ == "__main__":

    config_path_str = sys.argv[1] if len(sys.argv) > 1 else "configs/training/train_lam_cafca.yaml"
        
    cfg = OmegaConf.load(config_path_str)
    
    cli_overrides = OmegaConf.from_cli(sys.argv[2:])
    cfg = OmegaConf.merge(cfg, cli_overrides)
    
    logger.info("Configuration loaded:")
    logger.info(OmegaConf.to_yaml(cfg))
    train(cfg)

