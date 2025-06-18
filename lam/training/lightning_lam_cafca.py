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
from safetensors.torch import load_file
import torch
from time import time

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lam.models.modeling_lam import ModelLAM
from lam.losses import LPIPSLoss
from lam.training.train_lam_cafca import prepare_batch_for_model, get_logger

logger = get_logger(__name__)

class LamLightningModel(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(OmegaConf.to_container(cfg, resolve=True))
        self.cfg = cfg
        self.model = self._build_model(cfg)

        # Loss Functions
        self.l1_loss_fn = nn.L1Loss()
        # self.lpips_loss_fn = LPIPSLoss()

        self.time_metrics = {}

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

        # Fine-tuning: Freeze parameters if finetune_renderer_mlp_only is True
        if cfg.training.get("finetune_renderer_mlp_only", False):
            logger.info("Fine-tuning mode: Freezing all parameters except renderer.mlp_net.")
            for name, param in model.named_parameters():
                param.requires_grad = False
            
            if hasattr(model, 'renderer') and hasattr(model.renderer, 'mlp_net') and model.renderer.mlp_net is not None:
                for param in model.renderer.mlp_net.parameters():
                    param.requires_grad = True
                logger.info("Unfroze parameters of model.renderer.mlp_net.")
            else:
                logger.warning("model.renderer.mlp_net not found or is None. "
                               "No parameters specifically unfrozen for MLP fine-tuning. "
                               "Ensure model config `gs_mlp_network_config` is set if MLP is expected.")
        return model

    def forward(self, batch):
        model_input_data = prepare_batch_for_model(batch, self.device)
        return self.model(
            image=model_input_data["image"],
            source_c2ws=model_input_data["source_c2ws"],
            source_intrs=model_input_data["source_intrs"],
            render_c2ws=model_input_data["render_c2ws"],
            render_intrs=model_input_data["render_intrs"],
            flame_params=model_input_data["flame_params"],
            render_bg_colors=model_input_data["render_bg_colors"]
        )

    def training_step(self, batch, batch_idx):
        start_time = time()
        
        model_input_data = prepare_batch_for_model(batch, self.device)
        self.time_metrics['prepare_batch_time'] = time() - start_time
        
        start_time = time()

        model_output = self.model(
            image=model_input_data["image"],
            source_c2ws=model_input_data["source_c2ws"],
            source_intrs=model_input_data["source_intrs"],
            render_c2ws=model_input_data["render_c2ws"],
            render_intrs=model_input_data["render_intrs"],
            flame_params=model_input_data["flame_params"],
            latent_points=model_input_data.get("latent_points"),
            image_feats=model_input_data.get("image_feats"),
            render_bg_colors=model_input_data["render_bg_colors"]
        )
        self.time_metrics['model_forward_time'] = time() - start_time
        start_time = time()
        pred_rgb = model_output['comp_rgb']
        gt_rgb = model_input_data['gt_render_images']

        loss_l1 = self.l1_loss_fn(pred_rgb, gt_rgb)
        total_loss = self.cfg.training.l1_loss_weight * loss_l1
        self.time_metrics['loss_calculation_time'] = time() - start_time
        
        self.log_dict({
            'train/prepare_batch_time': self.time_metrics['prepare_batch_time'],
            'train/model_forward_time': self.time_metrics['model_forward_time'],
            'train/loss_calculation_time': self.time_metrics['loss_calculation_time']
        }, on_step=True, on_epoch=True)

        self.log('train/total_loss', total_loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('train/l1_loss', loss_l1, on_step=True, on_epoch=True)
        self.log('learning_rate', self.optimizers().param_groups[0]['lr'], on_step=True, on_epoch=False)

        if self.global_step % self.cfg.wandb.log_train_images_every_n_steps == 0 and self.logger is not None:
            self._log_image_samples(model_input_data, pred_rgb, gt_rgb, "train")

        return total_loss

    def validation_step(self, batch, batch_idx):
        model_input_data = prepare_batch_for_model(batch, self.device)

        model_output = self.model(
            image=model_input_data["image"],
            source_c2ws=model_input_data["source_c2ws"],
            source_intrs=model_input_data["source_intrs"],
            render_c2ws=model_input_data["render_c2ws"],
            render_intrs=model_input_data["render_intrs"],
            flame_params=model_input_data["flame_params"],
            latent_points=model_input_data.get("latent_points"),
            image_feats=model_input_data.get("image_feats"),
            render_bg_colors=model_input_data["render_bg_colors"]
        )
        pred_rgb = model_output['comp_rgb']
        gt_rgb = model_input_data['gt_render_images']

        loss_l1 = self.l1_loss_fn(pred_rgb, gt_rgb)
        total_loss = self.cfg.training.l1_loss_weight * loss_l1

        self.log('val/total_loss', total_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log('val/l1_loss', loss_l1, on_step=False, on_epoch=True)

        self._log_image_samples(model_input_data, pred_rgb, gt_rgb, "val")

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
            self.logger.experiment.log({f"{stage_prefix}/image_samples": [wandb.Image(grid)]}, step=self.global_step)


    def configure_optimizers(self):
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            logger.error("No trainable parameters found! Check model configuration and fine-tuning flags.")
            raise ValueError("No trainable parameters for the optimizer.")
        
        num_trainable_params = sum(p.numel() for p in trainable_params)
        logger.info(f"Number of trainable parameters: {num_trainable_params / 1e6:.2f}M")
        optimizer = optim.AdamW(trainable_params, lr=self.cfg.training.learning_rate)
        
        return optimizer