# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import os
import argparse
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
)
from lam.losses import LPIPSLoss, PixelLoss
import numpy as np
from PIL import Image
from accelerate.logging import get_logger
from .base_inferrer import Inferrer
from lam.runners import REGISTRY_RUNNERS
from safetensors.torch import load_file
from lam.dataset import env_paths
from lam.dataset.cafca_dataset import CafcaDataset

logger = get_logger(__name__)
def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to YAML config file')
    args, unknown = parser.parse_known_args()
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli(unknown)
    # If model_name is provided via CLI, set it in cfg and also update model.pretrained_model_name_or_path
    if 'model_name' in cli_cfg:
        cfg.model_name = cli_cfg.model_name
        if 'model' not in cfg:
            cfg.model = {}
        cfg.model.pretrained_model_name_or_path = cli_cfg.model_name
    return cfg


@REGISTRY_RUNNERS.register('infer.infer')
class LAMInferrer(Inferrer):

    EXP_TYPE: str = 'infer'

    def __init__(self):
        super().__init__()

        self.cfg = parse_configs()

        """
        configure_logger(
            stream_level=self.cfg.logger,
            log_level=self.cfg.logger,
        )
        """

        self.model: LAMInferrer = self._build_model(self.cfg).to(self.device)

        self.cafca_loader = None
        if self.cfg.get('use_cafca_dataset', False):
            logger.info("Initializing CafcaLamDataset for LAM inference.")
            if not hasattr(env_paths, 'subjects_train') or not env_paths.subjects_train:
                raise ValueError("env_paths.subjects_train is not defined or is empty. Please set it for CafcaLamDataset.")
            subject_id = self.cfg.get('cafca_subject_id_for_single_infer', None)
            self.cafca_dataset = CafcaDataset(subject_list=[subject_id], mode="lam_infer")
            self.cafca_loader = torch.utils.data.DataLoader(
                self.cafca_dataset, batch_size=1, shuffle=False, num_workers=0 # Batch size 1 for inference
            )
        self.l1_loss_fn = PixelLoss(option='l1')
        self.lpips_loss_fn = LPIPSLoss(device=self.device, prefetch=True)
        self.psnr_metric = PSNR(data_range=1.0).to(self.device)
        self.ssim_metric = SSIM(data_range=1.0).to(self.device)

    def _build_model(self, cfg):
        """
        from lam.models import model_dict
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model = hf_model_cls.from_pretrained(cfg.model_name)
        """
        from lam.models import ModelLAM
        model = ModelLAM(**cfg.model)

        resume = os.path.join(cfg.model.pretrained_model_name_or_path, "model.safetensors")
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

    def infer(self):
        # === NEW INFERENCE LOGIC (BATCHED) ===
        cfg = self.cfg
        if not hasattr(cfg, 'experiment') or not hasattr(cfg.experiment, 'subj_id'):
            raise RuntimeError(f"Config missing experiment.subj_id! Top-level keys: {list(cfg.keys())}")
        subj_id = int(cfg.experiment.subj_id)
        source_cam_ids = cfg.experiment.source_cam_ids
        assert len(source_cam_ids) == 2, "Exactly two source_cam_ids must be specified in config."

        from lam.dataset.cafca_lam_dataset import CafcaLamDataset
        dataset = CafcaLamDataset(subject_list=[subj_id], num_source_frames=2, num_driving_frames=0, image_size=cfg.training.image_size, is_val=True, mode="lam_infer")
        all_cam_ids = [item["cam_id"] for item in dataset.subject_data[subj_id]]
        driving_cam_ids = [cid for cid in all_cam_ids if cid not in source_cam_ids]

        # Get the two source views (as a dict)
        source_item = dataset.get_item_by_cam_ids(subj_id, source_cam_ids)
        # Prepare driving batch
        driving_imgs, driving_w2cs, driving_intrs, driving_masks, driving_flame_params = [], [], [], [], []
        for driving_meta in [item for item in dataset.subject_data[subj_id] if item["cam_id"] in driving_cam_ids]:
            driving_imgs.append(dataset._load_image_as_tensor(driving_meta["image_file_path"]))
            driving_w2cs.append(torch.from_numpy(driving_meta["world_2_cam_np"]).float())
            driving_intrs.append(torch.from_numpy(driving_meta["intrinsic_np"]).float())
            driving_masks.append(dataset._load_image_as_tensor(driving_meta["mask_file_path"]))
            flame_param = dataset._load_subject_flame_params(driving_meta["subject_flame_param_path"])
            # Expand flame params to [1, ...] for stacking
            for k in flame_param:
                flame_param[k] = flame_param[k].unsqueeze(0)
            driving_flame_params.append(flame_param)
        # Stack driving batch
        driving_imgs = torch.stack(driving_imgs)  # [N, C, H, W]
        driving_w2cs = torch.stack(driving_w2cs)  # [N, 4, 4]
        driving_intrs = torch.stack(driving_intrs)  # [N, 4, 4]
        driving_masks = torch.stack(driving_masks)  # [N, C, H, W]
        # Stack flame params
        flame_keys = driving_flame_params[0].keys()
        driving_flame_params_stacked = {k: torch.cat([fp[k] for fp in driving_flame_params], dim=0) for k in flame_keys}
        # Add batch dim if needed
        for k in driving_flame_params_stacked:
            driving_flame_params_stacked[k] = driving_flame_params_stacked[k].unsqueeze(0)  # [1, N, ...]
        # Fix betas shape to match training logic
        betas = driving_flame_params_stacked["betas"]
        if betas.ndim == 3:
            driving_flame_params_stacked["betas"] = betas[:, 0]
        # Model expects [B, N, ...], so add batch dim to driving views
        driving_imgs = driving_imgs.unsqueeze(0).to(self.device)  # [1, N, C, H, W]
        driving_w2cs = driving_w2cs.unsqueeze(0).to(self.device)  # [1, N, 4, 4]
        driving_intrs = driving_intrs.unsqueeze(0).to(self.device)  # [1, N, 4, 4]
        driving_masks = driving_masks.unsqueeze(0).to(self.device)  # [1, N, C, H, W]
        # Prepare sources
        latent_points = source_item["tokens"].to(self.device)  # [1, 2, ...]
        src_w2cs = source_item["source_w2cs"].to(self.device)  # [1, 2, 4, 4]
        src_intrs = source_item["source_intrs"].to(self.device)
        # Prepare render_bg_colors
        render_bg_colors = torch.ones((1, len(driving_cam_ids), 3), dtype=torch.float32, device=self.device)
        # Call model ONCE for all driving views
        with torch.no_grad():
            res = self.model(
                src_w2cs=src_w2cs,
                src_intrs=src_intrs,
                render_w2cs=driving_w2cs,
                render_intrs=driving_intrs,
                render_bg_colors=render_bg_colors,
                flame_params={k:v.to(self.device) for k,v in driving_flame_params_stacked.items()},
                latent_points=latent_points,
                image_feats=None
            )
            pred_rgbs = res["comp_rgb"].detach().cpu().numpy()[0]  # [N, H, W, 3]
            pred_rgbs = (np.clip(pred_rgbs, 0, 1.0) * 255).astype(np.uint8)
        # Save packed images for each driving view
        src1 = (source_item["source_rgbs"][0,0].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        src2 = (source_item["source_rgbs"][0,1].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
        H, W, _ = src1.shape

        # Ensure output directory exists
        if not hasattr(cfg, 'image_dump'):
            cfg.image_dump = './exps/inference_images_vs'
        os.makedirs(cfg.image_dump, exist_ok=True)

        for i, driving_cam_id in enumerate(driving_cam_ids):
            drv = (driving_imgs[0,i].cpu().numpy().transpose(1,2,0) * 255).astype(np.uint8)
            pred_rgb = pred_rgbs[i]
            # Ensure pred_rgb is [H, W, 3]
            if pred_rgb.shape[0] == 3 and pred_rgb.shape[-1] != 3:
                pred_rgb = np.transpose(pred_rgb, (1,2,0))
            # Resize all to match src1 if needed
            drv = np.array(Image.fromarray(drv).resize((W, H)))
            pred_rgb = np.array(Image.fromarray(pred_rgb).resize((W, H)))
            src2_resized = np.array(Image.fromarray(src2).resize((W, H)))
            packed = np.concatenate([src1, src2_resized, drv, pred_rgb], axis=1)
            packed_img = Image.fromarray(packed)
            packed_path = os.path.join(cfg.image_dump, f"subj{subj_id}_src{'-'.join(source_cam_ids)}_drv{driving_cam_id}.png")
            packed_img.save(packed_path)
            logger.info(f"Saved packed result to {packed_path}")
