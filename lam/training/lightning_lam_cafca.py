from multiprocessing.synchronize import BoundedSemaphore
import os
import sys
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
import torch.nn.functional as F
import wandb
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
)
import torch

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lam.models.modeling_lam import ModelLAM
from lam.losses import LPIPSLoss, PixelLoss
from lam.training.train_lam_cafca import prepare_batch_for_model, get_logger

logger = get_logger(__name__)

class LamLightningModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        if not isinstance(cfg, DictConfig):
            cfg = OmegaConf.create(cfg) 
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        self.model = self._build_model(cfg)

        # Loss Functions
        self.l1_loss_fn = nn.L1Loss()
        self.lpips_loss_fn = LPIPSLoss(device='cuda', prefetch=True)
        self.mask_loss_fn = PixelLoss(option="l1")
        self.psnr = PSNR(data_range=1.0)
        self.ssim = SSIM(data_range=1.0)

        
    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return prepare_batch_for_model(batch, device)

    def _build_model(self, cfg: DictConfig):
        model = ModelLAM(**cfg.model)
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
        # create perturbed mlps
        model.renderer.build_mlps()

        # Fine-tuning: Freeze parameters if finetune_renderer_mlp_only is True
        if cfg.training.get("finetune_renderer_mlp_only", False):
            print("Fine-tuning mode: Freezing all parameters except renderer.mlp_net.")
            for name, param in model.named_parameters():
                param.requires_grad = False
            
            # Handle both single and multiple MLPs
            if hasattr(model, 'renderer'):
                if hasattr(model.renderer, 'mlp_net') and model.renderer.mlp_net is not None:
                    # Single MLP case (num_gaussians_per_vertex == 1)
                    for param in model.renderer.mlp_net.parameters():
                        param.requires_grad = True
                    print("Unfroze parameters of model.renderer.mlp_net.")
                if hasattr(model.renderer, 'mlp_nets') and model.renderer.mlp_nets is not None:
                    # Multiple MLPs case (num_gaussians_per_vertex > 1)
                    for i, mlp_net in enumerate(model.renderer.mlp_nets):
                        for param in mlp_net.parameters():
                            param.requires_grad = True
                        print(f"Unfroze parameters of model.renderer.mlp_nets[{i}].")
            if hasattr(model, 'fusion_layer') and model.fusion_layer is not None:
                for param in model.fusion_layer.parameters():
                    param.requires_grad = True
                    # param.data.zero_()
                print("Unfroze parameters of model.fusion_layer.")
                print(param.data)
            if hasattr(model, 'renderer') and hasattr(model.renderer, 'gs_net') and model.renderer.gs_net is not None:
                for param in model.renderer.gs_net.parameters():
                    param.requires_grad = True
                print("Unfroze parameters of model.renderer.gs_net.")
            else:
                print("model.renderer.gs_net not found or is None. "
                               "No parameters specifically unfrozen for MLP fine-tuning. "
                               "Ensure model config `gs_mlp_network_config` is set if MLP is expected.")
            # ---------- NEW: unfreeze plücker fusion modules ------------------- #
            for attr in [
                'fusion_mlp',                    # two-layer MLP
                # 'post_fusion_transformer',    # cross-attention layer (commented out)
                'post_fusion_self_attention',    # self-attention layer
            ]:
                if hasattr(model, attr) and getattr(model, attr) is not None:
                    module = getattr(model, attr)
                    param_count = 0
                    for p in module.parameters():
                        p.requires_grad = True
                        param_count += 1
                    print(f"Unfroze {param_count} parameters of model.{attr}.")
                else:
                    print(f"model.{attr} not found or is None!")
            
            # Print parameter status AFTER unfreezing
            print("\n[DEBUG] Parameter status after unfreezing:")
            for name, param in model.named_parameters():
                if 'post_fusion_transformer' in name or 'fusion_mlp' in name or 'post_fusion_self_attention' in name:
                    print(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        
        # Count total model parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[DEBUG] Total model trainable parameters: {total_params:,}")
        
        # Count self-attention parameters if they exist
        if hasattr(model, 'post_fusion_self_attention') and model.post_fusion_self_attention is not None:
            sa_params = sum(p.numel() for p in model.post_fusion_self_attention.parameters() if p.requires_grad)
            print(f"[DEBUG] Self-attention layer trainable parameters: {sa_params:,}")
        
        # Count cross-attention parameters if they exist (commented out)
        # if hasattr(model, 'post_fusion_transformer') and model.post_fusion_transformer is not None:
        #     ca_params = sum(p.numel() for p in model.post_fusion_transformer.parameters() if p.requires_grad)
        #     print(f"[DEBUG] Cross-attention layer trainable parameters: {ca_params:,}")
        
        return model

    def forward(self, batch):
        # model_input_data = prepare_batch_for_model(batch, self.device)
        return self.model(
            src_w2cs=batch["src_w2cs"],
            src_intrs=batch["src_intrs"],
            render_w2cs=batch["render_w2cs"],
            render_intrs=batch["render_intrs"],
            flame_params=batch["flame_params"],
            # image_feats=batch.get("source_img_feats"),
            src_canon_2_cam=batch["source_canon_2_cam"],
            latent_points=batch["latent_points"],
            render_bg_colors=batch["render_bg_colors"]
        )

    def training_step(self, batch, batch_idx):
        bs = batch["render_w2cs"].size(0)
        model_output = self.model(
            src_w2cs=batch["src_w2cs"],
            src_intrs=batch["src_intrs"],
            render_w2cs=batch["render_w2cs"],
            render_intrs=batch["render_intrs"],
            flame_params=batch["flame_params"],
            # image_feats=batch.get("source_img_feats"),
            src_canon_2_cam=batch["source_canon_2_cam"],
            latent_points=batch["latent_points"],
            render_bg_colors=batch["render_bg_colors"]
        )

        pred_rgb = model_output['comp_rgb']
        gt_rgb = batch['gt_render_images']
        pred_mask  = model_output["comp_mask"]
        mask    = batch["driving_masks"]
        offset     = model_output["offset"]  

        loss_l1       = self.cfg.training.l1_loss_weight * self.l1_loss_fn(pred_rgb, gt_rgb)
        loss_lpips    = self.cfg.training.lpips_loss_weight * self.lpips_loss_fn(pred_rgb, gt_rgb)
        loss_mask     = self.cfg.training.silhouette_loss_weight * self.mask_loss_fn(pred_mask, mask)
        loss_offset   = self.cfg.training.offset_loss_weight *  (offset ** 2).mean()
        
        # SSIM needs 4D input: flatten [B, Nv, C, H, W] -> [B*Nv, C, H, W]
        B, Nv, C, H, W = pred_rgb.shape
        pred_flat = pred_rgb.reshape(B * Nv, C, H, W)
        gt_flat = gt_rgb.reshape(B * Nv, C, H, W)
        ssim_loss = self.cfg.training.get("ssim_loss_weight", 0.0) * (1.0 - self.ssim(pred_flat, gt_flat))
        
        total_loss = loss_l1 + loss_lpips  + loss_mask + loss_offset + ssim_loss

        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=False, batch_size=bs)
        self.log('train/l1_loss', loss_l1, on_step=True, on_epoch=False, batch_size=bs)
        self.log("train/lpips_loss", loss_lpips,   on_step=True, batch_size=bs)
        self.log("train/mask_loss",  loss_mask,    on_step=True, batch_size=bs)
        self.log("train/offset_loss",loss_offset,  on_step=True, batch_size=bs)
        self.log("train/ssim_loss", ssim_loss, on_step=True, batch_size=bs)
        # Log learning rates for different parameter groups
        optimizer = self.optimizers()
        for i, group in enumerate(optimizer.param_groups):
            group_name = group.get('name', f'group_{i}')
            self.log(f'learning_rate/{group_name}', group['lr'], on_step=True, on_epoch=False)

        if (self.global_step + 1) % self.trainer.num_training_batches == 0:
            self._log_image_samples(batch, pred_rgb, gt_rgb, "train")

        return total_loss

    def validation_step(self, batch, batch_idx):
        model_output = self.model(
            src_w2cs=batch["src_w2cs"],
            src_intrs=batch["src_intrs"],
            render_w2cs=batch["render_w2cs"],
            render_intrs=batch["render_intrs"],
            flame_params=batch["flame_params"],
            # image_feats=batch.get("source_img_feats"),
            src_canon_2_cam=batch["source_canon_2_cam"],
            latent_points=batch["latent_points"],
            render_bg_colors=batch["render_bg_colors"]
        )
        pred_rgb = model_output['comp_rgb']
        gt_rgb = batch['gt_render_images']
        pred_mask  = model_output["comp_mask"]
        driving_mask    = batch["driving_masks"]
        offset     = model_output["offset"]  

        loss_l1       = self.cfg.training.l1_loss_weight * self.l1_loss_fn(pred_rgb, gt_rgb)
        with torch.no_grad():
            loss_lpips = self.lpips_loss_fn(pred_rgb, gt_rgb)
        loss_lpips    *= self.cfg.training.lpips_loss_weight
        loss_mask     = self.cfg.training.silhouette_loss_weight * self.mask_loss_fn(pred_mask, driving_mask)
        loss_offset   = self.cfg.training.offset_loss_weight *  (offset ** 2).mean()
        
        # Flatten for SSIM and metrics
        B, Nv, C, H, W = pred_rgb.shape
        pred_flat = pred_rgb.reshape(B * Nv, C, H, W)
        gt_flat   = gt_rgb.reshape(B * Nv, C, H, W)
        
        ssim_loss = self.cfg.training.get("ssim_loss_weight", 0.0) * (1.0 - self.ssim(pred_flat, gt_flat))
        total_loss = loss_l1 + loss_lpips + loss_mask + loss_offset + ssim_loss

        psnr_val = self.psnr(pred_flat, gt_flat)
        ssim_val = self.ssim(pred_flat, gt_flat)

        self.log("val/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/l1_loss",    loss_l1,    on_step=False, on_epoch=True)
        self.log("val/lpips_loss", loss_lpips,   on_step=False, on_epoch=True)
        self.log("val/mask_loss",  loss_mask,    on_step=False, on_epoch=True)
        self.log("val/offset_loss",loss_offset,  on_step=False, on_epoch=True)

        self.log("val/ssim_loss", ssim_loss, on_step=False, on_epoch=True)
        self.log("val/psnr", psnr_val, on_step=False, on_epoch=True, batch_size=pred_flat.size(0))
        self.log("val/ssim", ssim_val, on_step=False, on_epoch=True, batch_size=pred_flat.size(0))
        if torch.rand(1).item() < 0.1:
            self._log_image_samples(batch, pred_rgb, gt_rgb, "val")

        return total_loss


    def _log_image_samples(self, model_input_data, pred_rgb, gt_rgb, stage_prefix="train"):
        num_samples_to_log_config = self.cfg.training.get(f"num_{stage_prefix}_samples_to_log", 1)
        actual_samples_to_log = min(num_samples_to_log_config, pred_rgb.shape[0])

        if actual_samples_to_log > 0 and self.logger and hasattr(self.logger.experiment, 'log'):
            log_preds = pred_rgb[:actual_samples_to_log].clamp(0, 1)
            log_gts = gt_rgb[:actual_samples_to_log].clamp(0, 1)
            log_srcs = model_input_data["image"][:actual_samples_to_log].clamp(0, 1)

            vis_images = []
            target_size = (self.cfg.training.image_size, self.cfg.training.image_size)

            for i in range(actual_samples_to_log):
                resized_srcs = F.interpolate(log_srcs[i], size=target_size, mode='bilinear', align_corners=False)
                vis_images.extend(list(resized_srcs))
                gt_view = log_gts[i, 0]
                pred_view = log_preds[i, 0]

                vis_images.append(F.interpolate(gt_view.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0))
                vis_images.append(F.interpolate(pred_view.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0))

            grid = vutils.make_grid(vis_images, nrow=log_srcs.shape[1] + 2, padding=2, normalize=False)
            self.logger.experiment.log({f"{stage_prefix}/image_samples": [wandb.Image(grid)]})


    def configure_optimizers(self):
        # Create parameter groups with different learning rates
        base_lr = self.cfg.training.learning_rate
        
        param_groups = []
        
        # Group 1: New cross-attention layer (higher LR since training from scratch) - COMMENTED OUT
        # if hasattr(self.model, 'post_fusion_transformer') and self.model.post_fusion_transformer is not None:
        #     ca_params = [p for p in self.model.post_fusion_transformer.parameters() if p.requires_grad]
        #     if ca_params:
        #         param_groups.append({
        #             'params': ca_params,
        #             'lr': base_lr * 2.0,  # 2x learning rate for new layer
        #             'name': 'cross_attention'
        #         })
        #         logger.info(f"Cross-attention parameters: {sum(p.numel() for p in ca_params) / 1e6:.2f}M (LR: {base_lr * 2.0})")
        
        # Group 1: New self-attention layer (higher LR since training from scratch)
        if hasattr(self.model, 'post_fusion_self_attention') and self.model.post_fusion_self_attention is not None:
            sa_params = [p for p in self.model.post_fusion_self_attention.parameters() if p.requires_grad]
            if sa_params:
                param_groups.append({
                    'params': sa_params,
                    'lr': base_lr,  # Much lower learning rate to prevent explosion
                    'name': 'self_attention'
                })
                logger.info(f"Self-attention parameters: {sum(p.numel() for p in sa_params) / 1e6:.2f}M (LR: {base_lr * 0.1})")
        
        # Group 2: Fusion MLP (medium-high LR)
        if hasattr(self.model, 'fusion_mlp') and self.model.fusion_mlp is not None:
            fusion_params = [p for p in self.model.fusion_mlp.parameters() if p.requires_grad]
            if fusion_params:
                param_groups.append({
                    'params': fusion_params,
                    'lr': base_lr * 1,  # 1.5x learning rate for fusion
                    'name': 'fusion_mlp'
                })
                logger.info(f"Fusion MLP parameters: {sum(p.numel() for p in fusion_params) / 1e6:.2f}M (LR: {base_lr * 1.5})")
        
        # Group 3: All other trainable parameters (base LR)
        other_params = []
        # Keep track of parameters already added to avoid duplicates
        added_params = set()
        
        # First, collect all parameters that are already in specific groups
        for group in param_groups:
            for param in group['params']:
                added_params.add(id(param))
        
        # Then add remaining trainable parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad and id(param) not in added_params:
                other_params.append(param)
        
        if other_params:
            param_groups.append({
                'params': other_params,
                'lr': base_lr,  # Base learning rate
                'name': 'other'
            })
            logger.info(f"Other parameters: {sum(p.numel() for p in other_params) / 1e6:.2f}M (LR: {base_lr})")
        
        if not param_groups:
            logger.error("No trainable parameters found! Check model configuration and fine-tuning flags.")
            raise ValueError("No trainable parameters for the optimizer.")
        
        total_params = sum(sum(p.numel() for p in group['params']) for group in param_groups)
        logger.info(f"Total trainable parameters: {total_params / 1e6:.2f}M")
        
        optimizer = optim.AdamW(param_groups)
        
        return optimizer