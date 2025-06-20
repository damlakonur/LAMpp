import os
import sys
import logging
import traceback
from pathlib import Path
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.utils as vutils
import wandb
from omegaconf import OmegaConf, DictConfig
from safetensors.torch import load_file
from tqdm import tqdm
import torch.nn.functional as F

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lam.dataset.cafca_lam_dataset import CafcaLamDataset
from lam.models.modeling_lam import ModelLAM
from lam.losses import LPIPSLoss

def get_logger(name, level=logging.INFO):
    """Initializes and returns a logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    if not logger.hasHandlers():
        logger.addHandler(handler)
    logger.propagate = False
    return logger

logger = get_logger(__name__)


def prepare_batch_for_model(batch, device):
    """
    Prepares a batch of data from CafcaLamDataset (already batched by DataLoader)
    for input to ModelLAM. Moves tensors to device and structures them as expected by the model.
    Casts image-related tensors to float32 to avoid dtype issues in torch.compile.
    """
    def _move(x, dtype=None):
        """Pinned-memory -> GPU async copy; optional fused cast."""
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype or x.dtype, non_blocking=True)
        return x
    prepared = {
        "image": _move(batch["source_rgbs"]),
        "latent_points": _move(batch["tokens"]),
    }
    
    prepared["render_c2ws"]   = _move(batch["driving_c2ws"])
    prepared["render_intrs"]  = _move(batch["driving_intrs"])
    prepared["render_bg_colors"] = _move(batch["render_bg_colors"])

    if "driving_masks" in batch:
        prepared["driving_masks"] = _move(batch["driving_masks"])

    prepared["gt_render_images"] = _move(batch["driving_image"])
    flame = {}
    betas = batch["betas"]
    if betas.ndim == 3:
        betas = betas[:, 0]
    flame["betas"] = _move(betas, torch.float32)

    for k in ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation"]:
        if k in batch:
            flame[k] = _move(batch[k], torch.float32)

    prepared["flame_params"] = flame
    if "uid" in batch:
        prepared["uid"] = batch["uid"] 

    # # Source data for encoding
    # prepared_batch["image"] = batch_from_dataloader["source_rgbs"].to(device).float()  # [B, N_ref, 3, H, W]

    # prepared_batch["latent_points"] = batch_from_dataloader["tokens"].to(device)

    # # Target/Driving data for rendering
    # prepared_batch["render_c2ws"] = batch_from_dataloader["driving_c2ws"].to(device).float()  # [B, N_render, 4, 4]
    # prepared_batch["render_intrs"] = batch_from_dataloader["driving_intrs"].to(device).float()  # [B, N_render, 4, 4]
    # prepared_batch["render_bg_colors"] = batch_from_dataloader["render_bg_colors"].to(device).float()  # [B, N_render, 3]
    # if "driving_masks" in batch_from_dataloader:
    #     prepared_batch["driving_masks"] = batch_from_dataloader["driving_masks"].to(device).float()

    # # Ground truth for loss
    # prepared_batch["gt_render_images"] = batch_from_dataloader["driving_image"].to(device).float()  # [B, N_render, 3, H, W]

    # # FLAME parameters
    # flame_params_for_model = {}
    # flame_keys_base = ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation"]

    # betas = batch_from_dataloader["betas"]
    # if betas.ndim == 3:  # [B, N_render, D]
    #     betas = betas[:, 0, :]
    # flame_params_for_model["betas"] = betas.to(device).float()  # [B, D]

    # for key in flame_keys_base:
    #     if key in batch_from_dataloader:
    #         flame_params_for_model[key] = batch_from_dataloader[key].to(device).float()

    # prepared_batch["flame_params"] = flame_params_for_model

    # # Optional UID
    # if "uid" in batch_from_dataloader:
    #     prepared_batch["uid"] = batch_from_dataloader["uid"]

    return prepared

def _build_model(cfg):
    """
    from lam.models import model_dict
    hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
    model = hf_model_cls.from_pretrained(cfg.model_name)
    """
    from lam.models import ModelLAM
    model = ModelLAM(**cfg.model)

    # resume = os.path.join(cfg.model_name, "model.safetensors")
    resume = os.path.join(cfg.experiment.model_name, "model.safetensors")
    print("==="*16*3)
    print("loading pretrained weight from:", resume)
    if resume.endswith('safetensors'):
        ckpt = load_file(resume, device='cpu')
    else:
        ckpt = torch.load(resume, map_location='cpu')
    state_dict = model.state_dict()
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
            else:
                print(f"WARN] mismatching shape for param {k}: ckpt {v.shape} != model {state_dict[k].shape}, ignored.")
        else:
            print(f"WARN] unexpected param {k}: {v.shape}")
    print("finish loading pretrained weight from:", resume)
    print("==="*16*3)
    return model

def train(cfg: DictConfig):
    """Main training loop."""
    exp_name = cfg.experiment.name
    output_dir = Path(cfg.experiment.output_dir) / exp_name / datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # WandB
    if cfg.wandb.enabled:
        wandb.init(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            name=cfg.wandb.get("run_name", exp_name),
            config=OmegaConf.to_container(cfg, resolve=True)
        )

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

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


    # Model
    logger.info("Initializing ModelLAM...")
    model = _build_model(cfg)
    
    model.to(device)

    # Fine-tuning: Freeze parameters if finetune_renderer_mlp_only is True
    if cfg.training.get("finetune_renderer_mlp_only", False):
        logger.info("Fine-tuning mode: Freezing all parameters except renderer.mlp_net.")
        for param in model.parameters():
            param.requires_grad = False
        
        if hasattr(model, 'renderer') and hasattr(model.renderer, 'mlp_net') and model.renderer.mlp_net is not None:
            for param in model.renderer.mlp_net.parameters():
                param.requires_grad = True
            logger.info("Unfroze parameters of model.renderer.mlp_net.")
        else:
            logger.warning("model.renderer.mlp_net not found or is None. "
                           "No parameters specifically unfrozen for MLP fine-tuning. "
                           "Ensure model config `gs_mlp_network_config` is set if MLP is expected.")
    
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        logger.error("No trainable parameters found! Check model configuration and fine-tuning flags.")
        return
    
    num_trainable_params = sum(p.numel() for p in trainable_params)
    logger.info(f"Number of trainable parameters: {num_trainable_params / 1e6:.2f}M")

    # Optimizer
    optimizer = optim.AdamW(trainable_params, lr=cfg.training.learning_rate)

    # Loss Functions
    l1_loss_fn = nn.L1Loss()
    # lpips_loss_fn = LPIPSLoss().to(device) # Uncomment if using LPIPS

    # Training Loop
    logger.info("Starting training...")
    global_step = 0
    for epoch in range(cfg.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        loop = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        for batch_idx, batch_data_from_loader in loop:
            optimizer.zero_grad()

            try:
                model_input_data = prepare_batch_for_model(batch_data_from_loader, device, cfg.model)
            except Exception as e:
                logger.error(f"Error in prepare_batch_for_model at epoch {epoch+1}, batch {batch_idx}: {e}")
                logger.error(traceback.format_exc())
                continue 

            # Model forward pass
            # ModelLAM.forward expects: image, source_c2ws, source_intrs, render_c2ws, render_intrs, flame_params, render_bg_colors
            model_output = model(
                image=model_input_data["image"],
                source_c2ws=model_input_data["source_c2ws"],
                source_intrs=model_input_data["source_intrs"],
                render_c2ws=model_input_data["render_c2ws"],
                render_intrs=model_input_data["render_intrs"],
                flame_params=model_input_data["flame_params"],
                render_bg_colors=model_input_data["render_bg_colors"]
            )
            pred_rgb = model_output['comp_rgb'] 
            gt_rgb = model_input_data['gt_render_images']

            # Calculate loss
            loss_l1 = l1_loss_fn(pred_rgb, gt_rgb)
            total_loss = cfg.training.l1_loss_weight * loss_l1
        
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            loop.set_postfix(loss=total_loss.item(), lr=optimizer.param_groups[0]["lr"])
            global_step += 1

            if cfg.wandb.enabled and global_step % cfg.wandb.log_every_n_steps == 0:
                log_dict = {
                    "train/total_loss": total_loss.item(),
                    "train/l1_loss": loss_l1.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr'],
                    "epoch": epoch + 1,
                    "global_step": global_step
                }

                wandb.log(log_dict)
            if cfg.wandb.enabled and global_step % cfg.wandb.log_train_images_every_n_steps == 0:
                num_train_samples_to_log_config = cfg.training.get("num_train_samples_to_log", 1)
                actual_samples_to_log = min(num_train_samples_to_log_config, pred_rgb.shape[0])
                
                if actual_samples_to_log > 0:
                    # Clamp predicted and GT images to [0,1]
                    log_preds = pred_rgb[:actual_samples_to_log].clamp(0, 1)         # [B, N_render, 3, H, W]
                    log_gts = gt_rgb[:actual_samples_to_log].clamp(0, 1)
                    log_srcs = model_input_data["image"][:actual_samples_to_log].clamp(0, 1)  # [B, N_src, 3, 504, 504]

                    vis_train = []
                    for i in range(actual_samples_to_log):
                        # Resize all source images to match target size (e.g., 512x512)
                        resized_srcs = F.interpolate(log_srcs[i], size=(512, 512), mode='bilinear', align_corners=False)  # [N_src, 3, 512, 512]
                        vis_train.extend(list(resized_srcs))  # Convert to list of [3, 512, 512] tensors

                        vis_train.append(F.interpolate(log_gts[i, 0].unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0))
                        vis_train.append(F.interpolate(log_preds[i, 0].unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0))

                    grid_train = vutils.make_grid(vis_train, nrow=log_srcs.shape[1] + 2, padding=2, normalize=False)
                    wandb.log({"train/image_samples": wandb.Image(grid_train)}, step=global_step)
                    
            logger.info(f"Epoch [{epoch+1}/{cfg.training.num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {total_loss.item():.4f}")

        avg_epoch_loss = epoch_loss / len(train_dataloader)
        logger.info(f"Epoch [{epoch+1}/{cfg.training.num_epochs}] completed. Average Training Loss: {avg_epoch_loss:.4f}")
        if cfg.wandb.enabled:
            wandb.log({"train/avg_epoch_loss": avg_epoch_loss, "epoch": epoch + 1})

        # Validation
        if val_dataloader and (epoch + 1) % cfg.training.get("validate_every_n_epochs", 10) == 0:
            model.eval()
            val_loss = 0.0
            logged_images = 0
            with torch.no_grad():
                for val_batch_idx, val_batch_data in enumerate(val_dataloader):
                    try:
                        val_model_input = prepare_batch_for_model(val_batch_data, device, cfg.model)
                        val_output = model(
                            image=val_model_input["image"],
                            source_c2ws=val_model_input["source_c2ws"],
                            source_intrs=val_model_input["source_intrs"],
                            render_c2ws=val_model_input["render_c2ws"],
                            render_intrs=val_model_input["render_intrs"],
                            flame_params=val_model_input["flame_params"],
                            render_bg_colors=val_model_input["render_bg_colors"]
                        )
                        val_pred_rgb = val_output['comp_rgb']
                        val_gt_rgb = val_model_input['gt_render_images']
                        
                        current_val_loss = l1_loss_fn(val_pred_rgb, val_gt_rgb)
                        val_loss += current_val_loss.item()

                        if cfg.wandb.enabled:
                            # Log N_target_views images from the first batch item
                            num_to_log_this_item = min(val_pred_rgb.shape[1], cfg.training.get("num_val_samples_to_log", 0) - logged_images)
                            if num_to_log_this_item > 0:
                                # Take first item in batch, and up to num_to_log_this_item driving views
                                log_preds = val_pred_rgb[0, :num_to_log_this_item].clamp(0,1)
                                log_gts = val_gt_rgb[0, :num_to_log_this_item].clamp(0,1)
                                log_srcs = val_model_input["image"][0].clamp(0,1)

                                vis_val = []
                                for i in range(actual_samples_to_log):
                                    # Resize all source images to match target size (e.g., 512x512)
                                    vis_val = F.interpolate(log_srcs[i], size=(512, 512), mode='bilinear', align_corners=False)  # [N_src, 3, 512, 512]
                                    vis_val.extend(list(resized_srcs))  # Convert to list of [3, 512, 512] tensors

                                    vis_val.append(F.interpolate(log_gts[i, 0].unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0))
                                    vis_val.append(F.interpolate(log_preds[i, 0].unsqueeze(0), size=(512, 512), mode='bilinear', align_corners=False).squeeze(0))

                                grid = vutils.make_grid(vis_val, nrow=log_srcs.shape[1] + 2, padding=2, normalize=False)
                                wandb.log({f"val/epoch_{epoch+1}_sample_{val_batch_idx}_view_{i}": wandb.Image(grid)})
                                
                    except Exception as e:
                        logger.error(f"Error during validation batch {val_batch_idx}: {e}")
                        logger.error(traceback.format_exc())
                        if "uid" in val_model_input: logger.error(f"Problematic val UIDs: {val_model_input['uid']}")
                        continue
            
            avg_val_loss = val_loss / len(val_dataloader)
            logger.info(f"Validation Epoch [{epoch+1}/{cfg.training.num_epochs}] completed. Average Validation Loss: {avg_val_loss:.4f}")
            if cfg.wandb.enabled:
                wandb.log({"val/avg_l1_loss": avg_val_loss, "epoch": epoch + 1})
        
        # Save checkpoint
        if (epoch + 1) % cfg.training.save_every_n_epochs == 0 or (epoch + 1) == cfg.training.num_epochs:
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch+1:04d}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            latest_checkpoint_path = checkpoint_dir / "latest.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': OmegaConf.to_container(cfg, resolve=True)
            }, latest_checkpoint_path)
            logger.info(f"Saved latest checkpoint to {latest_checkpoint_path}")


    logger.info("Training finished.")
    if cfg.wandb.enabled:
        wandb.finish()


if __name__ == "__main__":    
    config_path_str = sys.argv[1] if len(sys.argv) > 1 else "configs/training/train_lam_cafca.yaml"
        
    cfg = OmegaConf.load(config_path_str)
    
    cli_overrides = OmegaConf.from_cli(sys.argv[2:])
    cfg = OmegaConf.merge(cfg, cli_overrides)
    
    logger.info("Configuration loaded:")
    logger.info(OmegaConf.to_yaml(cfg))

    train(cfg)
