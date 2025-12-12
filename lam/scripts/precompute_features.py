import os
import sys
import logging
import traceback
from pathlib import Path
from safetensors.torch import load_file

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import numpy as np

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lam.dataset.cafca_dataset_precompute import CafcaDatasetPP
from lam.models.modeling_lam import ModelLAM
from lam.training.train_lam_cafca import get_logger as get_train_logger
from lam.dataset import env_paths

logger = get_train_logger(__name__)

def prepare_batch_for_model(batch_from_dataloader, device):
    """
    Prepares a batch of data from CafcaDatasetPP (already batched by DataLoader)
    for input to the feature precomputation model.
    """
    def _move(x, dtype=None):
        if torch.is_tensor(x):
            return x.to(device=device, dtype=dtype or x.dtype, non_blocking=True)
        return x 

    prepared_batch = {
        "image": _move(batch_from_dataloader["source_rgbs"]),
        "env_id": _move(batch_from_dataloader["env_id"]),
        "expr_id": _move(batch_from_dataloader["expr_id"]),
    }

    flame_params_for_model = {}
    flame_keys = ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation", "betas", "canon_2_cam"]

    for key in flame_keys:
        if key in batch_from_dataloader:
            param = batch_from_dataloader[key]
            # Ensure param is [B, D] by squeezing the middle dimension if it's [B, 1, D]
            if param.ndim == 3 and param.shape[1] == 1:
                param = param.squeeze(1)
            flame_params_for_model[key] = param.to(device).float()

    prepared_batch["flame_params"] = flame_params_for_model
    
    return prepared_batch

def _build_model(cfg: DictConfig):
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
        # if hasattr(model, 'renderer') and hasattr(model.renderer, 'gs_net') and model.renderer.gs_net is not None:
        #     for param in model.renderer.gs_net.parameters():
        #         param.requires_grad = True
        #     logger.info("Unfroze parameters of model.renderer.gs_net.")
        else:
            logger.warning("model.renderer.mlp_net not found or is None. "
                            "No parameters specifically unfrozen for MLP fine-tuning. "
                            "Ensure model config `gs_mlp_network_config` is set if MLP is expected.")
    return model

def precompute_and_save_batch(
    model: ModelLAM,
    batch_data_from_loader: dict,
    output_dir: Path,
    device: torch.device,
    batch_idx: int
):
    """
    Computes image_feats and tokens for a batch and saves them per sample
    into the respective subject's dataset directory.
    """

    try:
        model_input_data = prepare_batch_for_model(batch_data_from_loader, device)
    except Exception as e:
        logger.error(f"Error in prepare_batch_for_model for a batch (approx. index {batch_idx}): {e}")
        logger.error(traceback.format_exc())
        return

    image_for_encoder = model_input_data["image"][:, 0]
    flame_params_for_driving = model_input_data["flame_params"]

    query_points_for_tokens = None
    if model.latent_query_points_type.startswith("e2e_flame"):
        query_points_for_tokens, _, _, _ = model.renderer.get_query_points(
            flame_params_for_driving, device=device
        )

    tokens_batch, image_feats_batch = model.forward_latent_points(image_for_encoder, camera=None, query_points=query_points_for_tokens, additional_features={})

    subject_ids_int_tensor = batch_data_from_loader["subject_id_int_scalar"]
    source_cam_ids_list_of_lists = batch_data_from_loader["source_cam_ids_list_scalar"]

    # Save features for each sample in the batch
    for i in range(image_feats_batch.shape[0]):
        subject_id_int_sample = subject_ids_int_tensor[i].item()
        subject_id_str_sample = str(subject_id_int_sample).zfill(5)

        if not source_cam_ids_list_of_lists[i]:
            logger.warning(f"Sample {i} in batch (subject {subject_id_str_sample}) has no source camera ID. Skipping save for this sample.")
            continue
        source_camera_id_sample = source_cam_ids_list_of_lists[i][0]

        current_sample_subject_output_dir = (
            Path(env_paths.DATA_DIR)
            / subject_id_str_sample
            / f"env_{batch_data_from_loader['env_id'][i]}"
            / f"expr_{batch_data_from_loader['expr_id'][i]}"
            / "frames"
        )

        # image_feats_target_dir = current_sample_subject_output_dir / "image_feats"
        tokens_target_dir = current_sample_subject_output_dir / "tokens_20k"
        # image_feats_target_dir = current_sample_subject_output_dir / "image_feats"
        # image_feats_target_dir.mkdir(parents=True, exist_ok=True)
        tokens_target_dir.mkdir(parents=True, exist_ok=True)

        sanitized_camera_id = source_camera_id_sample.replace('/', '_')
        
        # sample_image_feat_path = image_feats_target_dir / f"{sanitized_camera_id}.npz"
        sample_token_path = tokens_target_dir / f"{sanitized_camera_id}.npz"

        # torch.save(image_feats_batch[i].cpu(), sample_image_feat_path)
        # torch.save(tokens_batch[i].cpu(), sample_token_path)
        # image_feat_np = image_feats_batch[i].detach().cpu().numpy().astype(np.float16)
        tokens_np = tokens_batch[i].detach().cpu().numpy().astype(np.float16)
        # np.savez_compressed(sample_image_feat_path, image_feats=image_feat_np)
        np.savez_compressed(sample_token_path, tokens=tokens_np)
        logger.debug(f"Saved features for subject {subject_id_str_sample}, camera {source_camera_id_sample} to {current_sample_subject_output_dir}")
    logger.info(f"Processed and saved features for batch {batch_idx}")


def precompute_features_main(cfg: DictConfig):
    logger.info(f"Precomputing features and saving into dataset structure under: {env_paths.DATA_DIR}")

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Initializing dataset for precomputation...")
    dataset = CafcaDatasetPP(
        subject_list=list(cfg.dataset.cafca_subject_ids_train),
        num_source_frames=cfg.dataset.num_of_src_views,
        image_size=cfg.training.image_size,
        is_val=False
    )
    print(f"Dataset size: {len(dataset)}.")
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=False
    )
    logger.info(f"Dataset size: {len(dataset)}. Dataloader size: {len(dataloader)} batches.")

    logger.info("Initializing ModelLAM for precomputation...")
    model = _build_model(cfg)
    model.to(device)

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Precomputing features")):
        precompute_and_save_batch(model, batch_data, Path("dummy"), device, batch_idx)

    logger.info("Feature precomputation finished.")

if __name__ == "__main__":
    config_path_str = sys.argv[1] if len(sys.argv) > 1 else "configs/training/precompute_lam_cafca.yaml"
    cfg = OmegaConf.load(config_path_str)
    cli_overrides = OmegaConf.from_cli(sys.argv[2:])
    cfg = OmegaConf.merge(cfg, cli_overrides)

    logger.info("Configuration loaded for precomputation:")
    logger.info(OmegaConf.to_yaml(cfg))

    precompute_features_main(cfg)