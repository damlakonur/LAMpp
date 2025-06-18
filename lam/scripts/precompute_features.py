import os
import sys
import logging
import traceback
from pathlib import Path
import datetime

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
import math

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lam.dataset.cafca_lam_dataset import CafcaLamDataset
from lam.models.modeling_lam import ModelLAM
from lam.training.train_lam_cafca import prepare_batch_for_model, get_logger as get_train_logger, _build_model
from lam.dataset import env_paths

logger = get_train_logger(__name__)

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
        query_points_for_tokens, _ = model.renderer.get_query_points(
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
            / f"{env_paths.EXPRESSION_ID}_{env_paths.ENVIRONMENT_ID}"
        )

        image_feats_target_dir = current_sample_subject_output_dir / "image_feats"
        tokens_target_dir = current_sample_subject_output_dir / "tokens"
        image_feats_target_dir.mkdir(parents=True, exist_ok=True)
        tokens_target_dir.mkdir(parents=True, exist_ok=True)

        sanitized_camera_id = source_camera_id_sample.replace('/', '_')
        
        sample_image_feat_path = image_feats_target_dir / f"{sanitized_camera_id}.pt"
        sample_token_path = tokens_target_dir / f"{sanitized_camera_id}.pt"

        torch.save(image_feats_batch[i].cpu(), sample_image_feat_path)
        torch.save(tokens_batch[i].cpu(), sample_token_path)
        logger.debug(f"Saved features for subject {subject_id_str_sample}, camera {source_camera_id_sample} to {current_sample_subject_output_dir}")
    logger.info(f"Processed and saved features for batch {batch_idx}")


def precompute_features_main(cfg: DictConfig):
    logger.info(f"Precomputing features and saving into dataset structure under: {env_paths.DATA_DIR}")

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    logger.info("Initializing dataset for precomputation...")
    dataset = CafcaLamDataset(
        subject_list=list(cfg.dataset.cafca_subject_ids_train),
        num_source_frames=cfg.dataset.num_of_src_views,
        num_driving_frames=cfg.dataset.num_of_target_views,
        image_size=cfg.training.image_size,
        is_val=False
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers,
        pin_memory=True
    )
    logger.info(f"Dataset size: {len(dataset)}. Dataloader size: {len(dataloader)} batches.")

    logger.info("Initializing ModelLAM for precomputation...")
    model = _build_model(cfg)
    model.to(device)

    for batch_idx, batch_data in enumerate(tqdm(dataloader, desc="Precomputing features")):
        precompute_and_save_batch(model, batch_data, Path("dummy"), device, batch_idx)

    logger.info("Feature precomputation finished.")

if __name__ == "__main__":
    config_path_str = sys.argv[1] if len(sys.argv) > 1 else "configs/training/train_lam_cafca.yaml"
    cfg = OmegaConf.load(config_path_str)
    cli_overrides = OmegaConf.from_cli(sys.argv[2:])
    cfg = OmegaConf.merge(cfg, cli_overrides)

    logger.info("Configuration loaded for precomputation:")
    logger.info(OmegaConf.to_yaml(cfg))

    precompute_features_main(cfg)