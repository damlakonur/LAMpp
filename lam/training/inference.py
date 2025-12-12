# run_infer_lam_cafca.py
import os, sys, csv, random, logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from torchvision.utils import make_grid
import torchvision
from PIL import Image
from itertools import combinations
import numpy as np
from tqdm.auto import tqdm
import torchvision.utils as vutils
from lam.utils.video import images_to_video
import torch.nn.functional as F
from lam.runners.infer.head_utils import prepare_motion_seqs
from mediapy import VideoWriter
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
import subprocess
import shutil


def add_audio_to_video(video_path: str, audio_source_path: str, output_path: str = None):
    """
    Copy audio from audio_source_path to video_path using ffmpeg.
    
    Args:
        video_path: Path to the video file (no audio or audio to replace)
        audio_source_path: Path to the audio source - can be a video file (.mp4) or audio file (.wav, .mp3, etc.)
        output_path: Output path for the video with audio. If None, replaces the original.
    """
    if not os.path.exists(audio_source_path):
        logger.warning(f"Audio source not found: {audio_source_path}, skipping audio merge")
        return False
    
    if not shutil.which("ffmpeg"):
        logger.warning("ffmpeg not found in PATH, skipping audio merge")
        return False
    
    if output_path is None:
        # Replace original: use temp file
        temp_output = str(video_path) + ".with_audio.mp4"
    else:
        temp_output = output_path
    
    try:
        # ffmpeg command to copy video from video_path and audio from audio_source_path
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),           # Input video (no audio)
            "-i", str(audio_source_path),    # Input audio source
            "-c:v", "copy",                  # Copy video codec (no re-encoding)
            "-c:a", "aac",                   # Encode audio as AAC
            "-map", "0:v:0",                 # Take video from first input
            "-map", "1:a:0",                 # Take audio from second input
            "-shortest",                     # Match shortest stream length
            temp_output
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.warning(f"ffmpeg failed: {result.stderr}")
            return False
        
        # If replacing original, move temp file to original path
        if output_path is None:
            os.replace(temp_output, str(video_path))
        
        logger.info(f"Added audio to video: {output_path or video_path}")
        return True
        
    except Exception as e:
        logger.warning(f"Failed to add audio: {e}")
        if os.path.exists(temp_output) and output_path is None:
            os.remove(temp_output)
        return False

# -----------------------------------------------------------------------------
# add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root)) if str(project_root) not in sys.path else None
# -----------------------------------------------------------------------------
from lam.dataset.cafca_lam_de_dataset_static import CafcaLamDataset
from lam.training.lightning_lam_cafca import LamLightningModel
from lam.losses import LPIPSLoss, PixelLoss
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
)


def load_checkpoint_with_shape_mismatch_handling(checkpoint_path, cfg):
    """
    Load checkpoint while handling shape mismatches for FLAME model parameters.
    """
    # Load checkpoint manually
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['state_dict']
    
    # Create model normally (let it load pretrained weights, we'll override them anyway)
    lit = LamLightningModel(cfg)
    model_state_dict = lit.state_dict()
    # Only load fine-tuned parameters (MLP, GS net, fusion MLP)
    # Skip frozen parameters (pcl_embed, flame_model, etc.)
    trainable_keys = ['renderer.mlp_net', 'renderer.gs_net', 'fusion_mlp']
    
    filtered_state_dict = {}
    loaded_count = 0
    skipped_count = 0
    
    for k, v in state_dict.items():
        # Check if this key corresponds to a trainable component
        is_trainable = any(key in k for key in trainable_keys)
        
        if not is_trainable:
            skipped_count += 1
            continue
            
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                filtered_state_dict[k] = v
                loaded_count += 1
                print(f"[INFO] Loaded {k} with shape {v.shape}")
            else:
                print(f"[WARN] Shape mismatch for {k}: ckpt {v.shape} != model {model_state_dict[k].shape}, parameter ignored")
        else:
            print(f"[WARN] Key {k} in checkpoint but not in model")
    
    print(f"\n{'='*60}")
    print(f"LOADING SUMMARY:")
    print(f"  Loaded finetuned params: {loaded_count}")
    print(f"  Skipped frozen params: {skipped_count}")
    print(f"{'='*60}\n")
    
    # Load the filtered state dict (your finetuned weights override everything)
    lit.load_state_dict(filtered_state_dict, strict=False)
    print(f"[INFO] Successfully loaded finetuned checkpoint from {checkpoint_path}")
    
    return lit

# -----------------------------------------------------------------------------
def get_logger(name="infer", level=logging.INFO):
    lg = logging.getLogger(name); lg.setLevel(level)
    if not lg.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s | %(message)s"))
        lg.addHandler(h)
    lg.propagate = False
    return lg

logger = get_logger("infer")

# -----------------------------------------------------------------------------
def seed_everything(seed: int = 12345):
    pl.seed_everything(seed, workers=True)

    def _worker_init(worker_id: int):
        random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        import numpy as np
        np.random.seed(seed + worker_id)
    return _worker_init, torch.Generator().manual_seed(seed)

# -----------------------------------------------------------------------------
def save_rgb_stack(rgb: torch.Tensor, out_dir: Path, basename: str,
                   cam_ids: List[str]):
    """
    rgb: [N, H, W, 3]  (0-1 float)
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    arr = (rgb.clamp(0,1)*255).cpu().numpy().astype("uint8")
    from PIL import Image
    for i in range(arr.shape[0]):
        dst = out_dir / f"{basename}_to_{cam_ids[i]}_{i:04d}.png"
        Image.fromarray(arr[i]).save(dst)
        
# -----------------------------------------------------------------------------
def override_driving_batch(batch, motion_seqs):
    """
    Overrides the driving parameters in the batch using motion_seqs output from prepare_motion_seqs().
    
    Args:
        batch (dict): Original batch from DataLoader (1 source view, 1 target view).
        motion_seqs (dict): Output of prepare_motion_seqs(). Contains:
            - "render_c2ws": [N, 4, 4]
            - "render_intrs": [N, 3, 3]
            - "render_bg_colors": [N, 3]
            - "flame_params": Dict[str, Tensor]
            - "vis_motion_render": [N, 3, H, W]
    
    Returns:
        dict: batch with driving params overridden to match full driven sequence.
    """
    device = batch["gt_render_images"].device
    new_batch = dict(batch)

    # Squeeze batch dimension from motion_seqs: [1, N, ...] -> [N, ...]
    c2ws = motion_seqs["render_c2ws"]         # [N, 4, 4]
    intrs = motion_seqs["render_intrs"]      # [N, 3, 3]
    bg_colors = motion_seqs["render_bg_colors"]  # [N, 3]
    K = new_batch["src_intrs"]
    B = 2
    last_color = bg_colors[:, -1:, :]        # shape [1,1,3]
    pad = last_color.repeat(1, 2, 1)         # shape [1,2,3]
    bg_colors_new = torch.cat([bg_colors, pad], dim=1)
    bottom_row = torch.tensor([0, 0, 0, 1], dtype=K.dtype, device=K.device)
    bottom_row = bottom_row.view(1, 1, 4).repeat(B, 1, 1)     # [B, 1, 4]


    # Pad K with an extra zero column: from 3×3 → 3×4
    K_3x4 = torch.cat([K, torch.zeros(B, 3, 1, device=K.device, dtype=K.dtype)], dim=2)

    # Final intrinsics 4×4 matrix
    K_4x4 = torch.cat([K_3x4, bottom_row], dim=1)


    # concat new_batch["src_w2cs"] and torch.inverse(c2ws)
    new_batch["render_w2cs"]     = torch.cat([new_batch["src_w2cs"].unsqueeze(0), torch.inverse(c2ws)], dim=1).to(device)   # [1, 2N, 4, 4]
    new_batch["render_intrs"]    = torch.cat([K_4x4.unsqueeze(0), intrs], dim=1).to(device)                 # [1, 2N, 3, 3] or [1, 2N, 4, 4]
    
    new_batch["render_bg_colors"] = bg_colors_new.to(device)
    new_batch["source_canon_2_cam"] = new_batch["source_canon_2_cam"].unsqueeze(0).to(device)
    new_batch["src_w2cs"] = new_batch["src_w2cs"].unsqueeze(0).to(device)
    new_batch["latent_points"] = new_batch["latent_points"].unsqueeze(0).to(device)

    # Override FLAME params (except betas)
    if "flame_params" in batch:
        for k, v in motion_seqs["flame_params"].items():
            if k == "betas" or k == "canon_2_cam":
                continue
            if k in batch["flame_params"]:
                if v.ndim == 2:
                    v = v.unsqueeze(0)  # [1, N, D]
                first_vec = v[:, 0:1, :]        # [1, 1, 100]
                pad = first_vec.repeat(1, 2, 1)  # repeat twice → [1, 2, 100]

                v_new = torch.cat([pad, v], dim=1) 
                new_batch["flame_params"][k] = v_new.to(device)
    return new_batch
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
def _stack_frame_list(ds, frames, want_mask=False):
    """Stack a list of frame-meta dicts into tensors."""
    rgbs  = torch.stack([ds._load_image_as_tensor(f["image_file_path"])
                         for f in frames])                      # [N,3,H,W]
    w2cs  = torch.stack([torch.from_numpy(f["world_2_cam_np"]).float()
                         for f in frames])                      # [N,4,4]
    intrs = torch.stack([torch.from_numpy(f["intrinsic_np"]).float()
                         for f in frames])                      # [N,4,4]


    return rgbs, w2cs, intrs
# -----------------------------------------------------------------------------
def build_source_dict(ds, subj_id, src_ids):
    """Return dict with the two source views (batch-dim already added)."""
    frames = [item for item in ds.subject_data[subj_id] if item["cam_id"] in src_ids]
    rgb, w2cs, intrs, _ = _stack_frame_list(ds, frames, want_mask=False)
    tokens = torch.stack([torch.from_numpy(np.load(f["token_file_path"])["tokens"])
                          for f in frames])
    return dict(
        source_rgbs   = rgb.unsqueeze(0),    # [1,N,3,H,W]
        source_w2cs   = w2cs.unsqueeze(0),   # [1,N,4,4]
        source_intrs  = intrs.unsqueeze(0),  # [1,N,4,4]
        tokens        = tokens.unsqueeze(0),  # [1,N,D]
    )
# -----------------------------------------------------------------------------
def build_driving_dict(ds, subj_id, drv_ids):
    """Build driving views dict with correct tensor shapes."""
    frames = [item for item in ds.subject_data[subj_id] if item["cam_id"] in drv_ids]
    drv_rgb, w2cs, intrs, masks = _stack_frame_list(ds, frames, want_mask=True)
    
    # Load FLAME params once and expand for all driving views
    flame_params = ds._load_subject_flame_params(frames[0]["subject_flame_param_path"])
    N = len(frames)
    
    # Handle betas specially - model expects [B,D] not [B,N,D]
    betas = flame_params.pop('betas')
    flame_params_expanded = {
        k: v.unsqueeze(0).repeat(N, 1) for k, v in flame_params.items()
    }
    flame_params_expanded['betas'] = betas  # Keep betas as [D]
    
    return dict(
        rgb   = drv_rgb.unsqueeze(0),          # [1,N,3,H,W]
        w2cs  = w2cs.unsqueeze(0),             # [1,N,4,4]
        intrs = intrs.unsqueeze(0),            # [1,N,4,4]
        masks = masks.unsqueeze(0) if masks is not None else None,
        flame_params = {k: v.unsqueeze(0) for k,v in flame_params_expanded.items()},  # Add batch dim
    )
# -----------------------------------------------------------------------------
@torch.no_grad()
def infer_all_pairs(cfg, source_cam_ids=None, driving_env_id=None, driving_expr_id=None):
    dev = torch.device("cuda")
    subj = int(cfg.experiment.subj_id)

    # Initialize dataset
    ds = CafcaLamDataset(
        [subj], 
        num_source_frames=cfg.model.num_source_views,  # Use model's expected number of source views
        num_driving_frames=1,  # Set to 1 for this inference case
        image_size=cfg.training.image_size,
        is_val=True,
        max_cache_size=cfg.dataset.get("max_cache_size", 128)
    )

    # Get source environment and expression IDs from config
    source_env_id = cfg.experiment.get("env_id")
    source_expr_id = cfg.experiment.get("expr_id")
    
    # Driving env/expr must be provided as function arguments
    if driving_env_id is None or driving_expr_id is None:
        raise ValueError("driving_env_id and driving_expr_id must be provided as function arguments")

    # Look up source expression in the new dataset structure
    source_expr_key = (subj, source_env_id, source_expr_id)
    if source_expr_key not in ds.expression_data:
        raise ValueError(f"Source expression {source_expr_key} not found in dataset")
    
    source_expr_info = ds.expression_data[source_expr_key]
    source_frames = source_expr_info['frames']

    # Look up driving expression in the new dataset structure
    driving_expr_key = (subj, driving_env_id, driving_expr_id)
    if driving_expr_key not in ds.expression_data:
        raise ValueError(f"Driving expression {driving_expr_key} not found in dataset")
    
    driving_expr_info = ds.expression_data[driving_expr_key]
    driving_frames = driving_expr_info['frames']

    # Source candidates must be provided via source_cam_ids
    # Use model's num_source_views, not dataset's num_of_src_views, since the model architecture is fixed
    num_src_views = cfg.model.num_source_views
    if source_cam_ids is None or len(source_cam_ids) == 0:
        raise ValueError("Must provide at least 1 source camera ID")
    if len(source_cam_ids) > num_src_views:
        raise ValueError(f"Cannot provide more than {num_src_views} source camera IDs (got {len(source_cam_ids)}). Model was trained with {num_src_views} source views.")
    
    # Map camera IDs to dataset indices within source frames
    cam_id_to_idx = {frame["cam_id"]: idx for idx, frame in enumerate(source_frames)}
    source_candidates = [cam_id_to_idx[cam_id] for cam_id in source_cam_ids if cam_id in cam_id_to_idx]
    if len(source_candidates) != len(source_cam_ids):
        missing_cams = [cam_id for cam_id in source_cam_ids if cam_id not in cam_id_to_idx]
        raise ValueError(f"Could not find source camera IDs {missing_cams} in source frames")
    
    # Load model
    ckpt = cfg.experiment.get("checkpoint", None)
    if ckpt and Path(ckpt).exists():
        # lit = load_checkpoint_with_shape_mismatch_handling(ckpt, cfg)
        lit = LamLightningModel.load_from_checkpoint(ckpt)
        logger.info(f"Loaded weights from {ckpt}")
    else:
        lit = LamLightningModel(cfg)
        logger.info(f"Loaded base model from {cfg.experiment.model_name}")
    lit.eval().cuda()

    # Setup metrics
    psnr = PSNR(data_range=1.0).to(dev)
    ssim = SSIM(data_range=1.0).to(dev)
    lpips = LPIPSLoss(device=dev, prefetch=True)
    l1_fn = torch.nn.L1Loss()

    # Create output directory
    dir_name = "images" + "_" + str(cfg.experiment.get("subj_id", 0)) + "_" + driving_expr_id
    out_root = Path(cfg.experiment.output_dir) / dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Use the exact number of source frames provided
    src_frames = [source_frames[idx] for idx in source_candidates]
    src_ids = [f["cam_id"] for f in src_frames]
    src_flame_params = ds._load_subject_flame_params(source_expr_info['flame_params_path'])
    betas = src_flame_params['betas'].clone()  # Keep original betas
    canon_2_cam = src_flame_params['canon_2_cam'].repeat(len(src_frames), 1, 1)
    
    # Use ALL driving frames from the specified driving env/expr (all ~30 camera views)
    drv_frames = driving_frames
    drv_ids = [f["cam_id"] for f in drv_frames]
    
    # Build source dict directly from our filtered source frames
    rgb, w2cs, intrs = _stack_frame_list(ds, src_frames)
    tokens = torch.stack([ds._get_token_tensor(f["token_file_path"]) for f in src_frames])
    
    src = dict(
        source_rgbs   = rgb.unsqueeze(0),    # [1,N,3,H,W]
        source_w2cs   = w2cs.unsqueeze(0),   # [1,N,4,4]
        source_intrs  = intrs.unsqueeze(0),  # [1,N,4,4]
        tokens        = tokens.unsqueeze(0),  # [1,N,D]
        canon_2_cam   = canon_2_cam.unsqueeze(0),  # [1,N,4,4]
    )

    # Build driving dict directly from our filtered driving frames
    drv_rgb, drv_w2cs, drv_intrs = _stack_frame_list(ds, drv_frames, want_mask=True)
    
    # Load FLAME params once and expand for all driving views
    flame_params = ds._load_subject_flame_params(driving_expr_info['flame_params_path'])
    N = len(drv_frames)
    
    # Handle betas specially - model expects [B,D] not [B,N,D]
    flame_params_expanded = {}
    for k, v in flame_params.items():
        if k == 'betas':
            flame_params_expanded[k] = v  # Keep betas as [D] (will be expanded later)
        elif v.dim() == 1:
            flame_params_expanded[k] = v.unsqueeze(0).repeat(N, 1)  # [D] -> [N, D]
        elif v.dim() == 2:
            flame_params_expanded[k] = v.unsqueeze(0).repeat(N, 1, 1)  # [D1, D2] -> [N, D1, D2]
        else:
            flame_params_expanded[k] = v.unsqueeze(0).expand(N, *v.shape)  # General case
    
    drv = dict(
        rgb   = drv_rgb.unsqueeze(0),          # [1,N,3,H,W]
        w2cs  = drv_w2cs.unsqueeze(0),             # [1,N,4,4]
        intrs = drv_intrs.unsqueeze(0),            # [1,N,4,4]
        # masks = masks.unsqueeze(0) if masks is not None else None,
        flame_params = {k: v.unsqueeze(0) for k,v in flame_params_expanded.items()},  # Add batch dim
    )

    # Prepare model input
    batch = {
        "src_w2cs": src["source_w2cs"].to(dev),
        "src_intrs": src["source_intrs"].to(dev),
        "source_canon_2_cam": src["canon_2_cam"].to(dev),
        "render_w2cs": drv["w2cs"].to(dev),
        "render_intrs": drv["intrs"].to(dev),
        "render_bg_colors": torch.ones((1, len(drv_ids), 3), device=dev),
        "flame_params": {k: v.to(dev) for k, v in drv["flame_params"].items()},
        "latent_points": src["tokens"].to(dev).float(),  # Convert from float16 to float32
    }

    # Forward pass
    out = lit(batch)
    pred = out["comp_rgb"][0]  # [D,3,H,W]
    gt = drv["rgb"][0].to(dev)  # [D,3,H,W]

    # Initialize metrics list
    metrics = []

    # Save visualization for each driving view (all ~30 views)
    for k, drv_id in enumerate(drv_ids):
        # Create horizontal layout: [src1, src2, ..., srcN, gt, pred]
        vis_tensors = []
        # Add all source views
        for i in range(len(src_frames)):
            vis_tensors.append(src["source_rgbs"][0, i].cpu())
        # Add ground truth and prediction
        vis_tensors.extend([
            gt[k].cpu(),                     
            pred[k].cpu()                    
        ])
        
        # Concatenate horizontally and save
        vis = torch.cat(vis_tensors, dim=2)  # [3,H,W*(N+2)] where N is num source views
        vis_np = (vis.clamp(0, 1) * 255).byte().numpy()
        vis_np = np.transpose(vis_np, (1, 2, 0))  # [H,W*(N+2),3]
        
        fn = out_root / f"subj{subj}_src{'-'.join(src_ids)}_drv{drv_id}_20k.png"
        Image.fromarray(vis_np).save(fn)
        pred_np = (pred[k].cpu().clamp(0, 1) * 255).byte().numpy()
        pred_np = np.transpose(pred_np, (1, 2, 0))  # [H,W,3]
        
        fn_pred = out_root / f"subj{subj}_src{'-'.join(src_ids)}_drv{drv_id}_20k_pred.png"
        Image.fromarray(pred_np).save(fn_pred)

        # Compute metrics
        l1 = l1_fn(pred[k], gt[k]).item()
        lp = lpips(pred[k:k+1], gt[k:k+1]).item()
        ps = psnr(pred[k:k+1], gt[k:k+1]).item()
        ss = ssim(pred[k:k+1], gt[k:k+1]).item()
        
        # Store metrics
        metrics.append({
            'source_pair': '-'.join(src_ids),
            'driving_view': drv_id,
            'l1': l1,
            'lpips': lp,
            'psnr': ps,
            'ssim': ss
        })
        
        print(f"src {src_ids} → drv {drv_id} | "
              f"L1 {l1:.3f} LPIPS {lp:.3f} PSNR {ps:.2f} SSIM {ss:.3f}")

    # Compute and log averages
    avg_l1 = np.mean([m['l1'] for m in metrics])
    avg_lpips = np.mean([m['lpips'] for m in metrics])
    avg_psnr = np.mean([m['psnr'] for m in metrics])
    avg_ssim = np.mean([m['ssim'] for m in metrics])
    
    print(f"\nsrc {src_ids} AVERAGE over {len(drv_ids)} views | "
          f"L1 {avg_l1:.3f} LPIPS {avg_lpips:.3f} PSNR {avg_psnr:.2f} SSIM {avg_ssim:.3f}")

    # Write metrics to CSV file
    csv_file = out_root / f"subj{subj}_src{'-'.join(src_ids)}_drv{driving_env_id}_{driving_expr_id}_metrics.csv"
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_pair', 'driving_view', 'l1', 'lpips', 'psnr', 'ssim'])
        for m in metrics:
            writer.writerow([m['source_pair'], m['driving_view'], m['l1'], m['lpips'], m['psnr'], m['ssim']])
        # Add average row
        writer.writerow(['-'.join(src_ids), 'AVERAGE', avg_l1, avg_lpips, avg_psnr, avg_ssim])

    logger.info(f"Completed rendering for {len(drv_ids)} driving views with source pair {src_ids}")
    logger.info(f"Metrics saved to {csv_file}")

@torch.no_grad()
def infer(cfg: DictConfig):
    seed_worker, torch_gen = seed_everything(12345)

    # ---------- Dataset ----------
    dataset = CafcaLamDataset(
        subject_list = list(cfg.dataset.cafca_subject_ids_val),
        num_source_frames = cfg.dataset.num_of_src_views,
        num_driving_frames = cfg.dataset.num_of_target_views,
        image_size = cfg.training.image_size,
        is_val = True,
        max_cache_size = cfg.dataset.get("max_cache_size", 128)  # Updated parameter name and default value
    )
    loader = DataLoader(dataset,
                        batch_size   = 1,
                        shuffle      = False,
                        num_workers  = 2,
                        worker_init_fn = seed_worker,
                        generator    = torch_gen)

    logger.info(f"Inference set: {len(dataset)} frames")

    # ---------- Lightning model ----------
    ckpt = cfg.experiment.get("checkpoint", None)
    if ckpt and Path(ckpt).exists():
        lit = load_checkpoint_with_shape_mismatch_handling(ckpt, cfg)
        logger.info(f"Loaded weights from {ckpt}")

    lit.eval().cuda()

    # Metrics helpers
    lpips = LPIPSLoss(device="cuda", prefetch=True)
    psnr  = PSNR(data_range=1.0).cuda()
    ssim  = SSIM(data_range=1.0).cuda()
    l1_fn = torch.nn.L1Loss()

    out_root = Path(cfg.experiment.output_dir)
    (out_root / "images").mkdir(parents=True, exist_ok=True)

    # ---------- Loop ----------
    with torch.no_grad():
        for n, raw_batch in enumerate(tqdm(loader, desc="infer")):
            batch = lit.transfer_batch_to_device(raw_batch, device="cuda", dataloader_idx=0)
            out = lit(batch)

            rgb = out["comp_rgb"]  # [1, Nv, 3, H, W]
            
            B, Nv, C, H, W = rgb.shape

            uid = batch["uid"][0] if isinstance(batch["uid"], list) else batch["uid"]
            dst_dir = out_root / "images" / uid
            dst_dir.mkdir(parents=True, exist_ok=True)

            # Input tensors
            src_rgb = batch["image"][0].clamp(0, 1)            # [Ns, 3, H, W]
            gt_rgb  = batch["gt_render_images"][0].clamp(0, 1) # [Nv, 3, H, W]
            pred_rgb = rgb[0].clamp(0, 1) 

            target_size = (cfg.training.image_size, cfg.training.image_size)
            Ns = src_rgb.shape[0]

            for i in range(Nv):
                vis_images = []

                # Resize source views
                for j in range(Ns):
                    src_img = src_rgb[j].unsqueeze(0)  # [1, 3, H, W]
                    resized = F.interpolate(src_img, size=target_size, mode='bilinear', align_corners=False)
                    vis_images.append(resized.squeeze(0))  # [3, H, W]

                # Resize GT and prediction
                gt_img = gt_rgb[i].unsqueeze(0)  # [1, 3, H, W]
                gt_resized = F.interpolate(gt_img, size=target_size, mode='bilinear', align_corners=False).squeeze(0)

                pred_img = pred_rgb[i].unsqueeze(0)  # [1, 3, H, W]
                pred_resized = F.interpolate(pred_img, size=target_size, mode='bilinear', align_corners=False).squeeze(0)

                vis_images.extend([gt_resized, pred_resized])

                # Build grid and save
                grid = vutils.make_grid(vis_images, nrow=len(vis_images), padding=2, normalize=False)
                grid_np = (grid * 255).permute(1, 2, 0).byte().cpu().numpy()
                fn = f"{uid}_viz_{i:04d}.png"
                Image.fromarray(grid_np).save(dst_dir / fn)

            # ---------- Metrics ----------
            l1 = l1_fn(pred_rgb, gt_rgb).item()
            lp = lpips(pred_rgb.contiguous(), gt_rgb).item()
            ps = psnr(pred_rgb, gt_rgb)
            ss = ssim(pred_rgb, gt_rgb)

            csv_path = dst_dir / "metrics.csv"
            if not csv_path.exists():
                with open(csv_path, "w") as f:
                    csv.writer(f).writerow(["l1", "lpips", "psnr", "ssim"])
            with open(csv_path, "a") as f:
                csv.writer(f).writerow([l1, lp, ps.item(), ss.item()])

@torch.no_grad()               
def infer_video_from_first_batch(cfg: DictConfig):
    seed_worker, torch_gen = seed_everything(12345)

    # ---------- Dataset ----------
    dataset = CafcaLamDataset(
        subject_list = list(cfg.dataset.cafca_subject_ids_val),
        num_source_frames = cfg.dataset.num_of_src_views,
        num_driving_frames = cfg.dataset.num_of_target_views,
        image_size = cfg.training.image_size,
        is_val = True,
        max_cache_size = cfg.dataset.get("max_cache_size", 128)  # Updated parameter name and default value
    )
    loader = DataLoader(dataset,
                        batch_size   = 1,
                        shuffle      = False,
                        num_workers  = 2,
                        worker_init_fn = seed_worker,
                        generator    = torch_gen)

    logger.info(f"Inference set: {len(dataset)} frames")

    # ---------- Lightning model ----------
    ckpt = cfg.experiment.get("checkpoint", None)
    if ckpt and Path(ckpt).exists():
        lit = load_checkpoint_with_shape_mismatch_handling(ckpt, cfg)
        logger.info(f"Loaded weights from {ckpt}")

    lit.eval().cuda()


    out_root = Path(cfg.experiment.output_dir)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    video_root = out_root / "videos"
    video_root.mkdir(parents=True, exist_ok=True)



    raw_batch = next(iter(loader))  # take first sample only
    breakpoint()
    
    batch = lit.transfer_batch_to_device(raw_batch, device="cuda", dataloader_idx=0)
    breakpoint()

    betas = raw_batch["betas"]

    if betas.ndim == 3:
        betas = betas[:, 0]
    shape_param = betas[0:1].to("cuda")  # [1, num_shape_coeffs] - keep batch dim for render_flame_mesh

    # ---------- Motion sequence generation ----------
    dump_dir = Path(cfg.experiment.output_dir) / "videos"
    dump_dir.mkdir(parents=True, exist_ok=True)
    dump_video_path = video_root / f"{batch['uid'][0]}_driven_motion_video.mp4"


    motion_seqs = prepare_motion_seqs(
        motion_seqs_dir    = cfg.experiment.get("motion_seqs_dir", None),
        image_folder       = None,
        save_root          = dump_dir,
        fps                = 30,
        bg_color           = 1,
        aspect_standard    = 1.0,
        enlarge_ratio      = [1.0, 1.0],
        render_image_res   = 512,
        need_mask          = False,
        multiply           = 16,
        vis_motion         = False,
        shape_param        = shape_param,
    )
    motion_seqs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in motion_seqs.items()}
    new_batch = override_driving_batch(batch, motion_seqs)
    out = lit(new_batch)

    rgb = out["comp_rgb"].detach().squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # [Nv, H, W, 3]
    rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

    images_to_video(rgb, output_path=str(dump_video_path), fps=30, gradio_codec=False, verbose=True)
    logger.info(f"Saved driven video to {dump_video_path}")
    
    
@torch.no_grad()               
def infer_video(cfg: DictConfig):

    # ---------- Dataset ----------
    dataset = CafcaLamDataset(
        subject_list = list(cfg.dataset.cafca_subject_ids_val),
        num_source_frames = cfg.dataset.num_of_src_views,
        num_driving_frames = cfg.dataset.num_of_target_views,
        image_size = cfg.training.image_size,
        is_val = True,
        max_cache_size = cfg.dataset.get("max_cache_size", 128)  # Updated parameter name and default value
    )

    logger.info(f"Inference set: {len(dataset)} subjects")

    # ---------- Lightning model ----------
    # ckpt = cfg.experiment.get("checkpoint", None)
    # if ckpt and Path(ckpt).exists():
    #     lit = LamLightningModel.load_from_checkpoint(ckpt)
    #     logger.info(f"Loaded weights from {ckpt}")
    
    finetuned_ckpt = cfg.experiment.checkpoint
    assert finetuned_ckpt and Path(finetuned_ckpt).exists(), "Checkpoint not found"
    # lit = load_checkpoint_with_shape_mismatch_handling(finetuned_ckpt, cfg)
    # TODO: check here
    lit = LamLightningModel.load_from_checkpoint(finetuned_ckpt)
    lit.eval().cuda()

    out_root = Path(cfg.experiment.output_dir)
    video_name = "videos" + "_" + str(cfg.experiment.get("subj_id", 0))
    video_root = out_root / video_name
    video_root.mkdir(parents=True, exist_ok=True)

    # ---------- Manually specify source cam IDs ----------
    subj_id = cfg.experiment.get("subj_id", 0)
    source_cam_ids = cfg.experiment.get("source_cam_ids", ["C0"])
    raw_batch = dataset.get_item_by_cam_ids(subj_id, source_cam_ids, cfg.experiment.get("env_id", "002"), cfg.experiment.get("expr_id", "expr_00009"))
    batch = lit.transfer_batch_to_device(raw_batch, device="cuda", dataloader_idx=0)

    betas = raw_batch["betas"]
    if betas.ndim == 3:
        betas = betas[:, 0]
    shape_param = betas[0:1].to("cuda")  # [1, num_shape_coeffs] - keep batch dim for render_flame_mesh

    # ---------- Motion sequence generation ----------
    dump_video_path = video_root / f"{batch['uid'][0]}_driven_motion_video.mp4"

    export_motion_video = cfg.experiment.get("export_motion_video", False)
    motion_seqs = prepare_motion_seqs(
        motion_seqs_dir    = cfg.experiment.get("motion_seqs_dir", None),
        image_folder       = None,
        save_root          = video_root,
        fps                = 30,
        bg_color           = 1,
        aspect_standard    = 1.0,
        enlarge_ratio      = [1.0, 1.0],
        render_image_res   = 512,
        need_mask          = False,
        multiply           = 16,
        vis_motion         = export_motion_video,
        shape_param        = shape_param,
    )
    motion_seqs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in motion_seqs.items()}

    # Store flame mesh frames if exporting motion video
    flame_mesh_frames = None
    if export_motion_video:
        flame_mesh_frames = motion_seqs["vis_motion_render"]  # [N, H, W, 3] uint8

    new_batch = override_driving_batch(batch, motion_seqs)
    out = lit(new_batch)

    rgb = out["comp_rgb"].detach().squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # [Nv, H, W, 3]
    rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
    # save images
    for i in range(rgb.shape[0]):
        save_file = os.path.join(video_root, f"{i:04d}.png")
        Image.fromarray(rgb[i]).save(save_file)

    images_to_video(rgb, output_path=str(dump_video_path), fps=30, gradio_codec=False, verbose=True)
    logger.info(f"Saved driven video to {dump_video_path}")

    # Add audio from original driving video/audio file if specified
    audio_source = cfg.experiment.get("audio_source", None) or cfg.experiment.get("audio_source_video", None)
    if audio_source and os.path.exists(audio_source):
        add_audio_to_video(str(dump_video_path), audio_source)

    # Save side-by-side comparison video (RGB | FLAME mesh)
    if export_motion_video and flame_mesh_frames is not None:
        # Save FLAME mesh video separately
        flame_mesh_video_path = video_root / f"{batch['uid'][0]}_flame_mesh_video.mp4"
        flame_mesh_dir = video_root / "flame_mesh_frames"
        flame_mesh_dir.mkdir(parents=True, exist_ok=True)
        for i in range(flame_mesh_frames.shape[0]):
            Image.fromarray(flame_mesh_frames[i]).save(flame_mesh_dir / f"flame_{i:04d}.png")
        images_to_video(flame_mesh_frames, output_path=str(flame_mesh_video_path), fps=30, gradio_codec=False, verbose=True)
        logger.info(f"Saved FLAME mesh animation video to {flame_mesh_video_path}")

        # Create side-by-side comparison video
        # RGB frames start from index 2 (first 2 are source views in override_driving_batch)
        rgb_for_comparison = rgb[2:]  # Skip the first 2 source view renders
        
        # Make sure both have the same number of frames
        num_frames = min(len(rgb_for_comparison), len(flame_mesh_frames))
        rgb_for_comparison = rgb_for_comparison[:num_frames]
        flame_mesh_for_comparison = flame_mesh_frames[:num_frames]
        
        # Resize if dimensions don't match
        H_rgb, W_rgb = rgb_for_comparison.shape[1:3]
        H_mesh, W_mesh = flame_mesh_for_comparison.shape[1:3]
        
        if H_rgb != H_mesh or W_rgb != W_mesh:
            # Resize mesh frames to match RGB
            resized_mesh = []
            for frame in flame_mesh_for_comparison:
                resized = np.array(Image.fromarray(frame).resize((W_rgb, H_rgb), Image.LANCZOS))
                resized_mesh.append(resized)
            flame_mesh_for_comparison = np.stack(resized_mesh)
        
        # Concatenate horizontally: [RGB | FLAME mesh]
        side_by_side = np.concatenate([rgb_for_comparison, flame_mesh_for_comparison], axis=2)  # [N, H, W*2, 3]
        
        # Save side-by-side video
        comparison_video_path = video_root / f"{batch['uid'][0]}_rgb_mesh_comparison.mp4"
        comparison_dir = video_root / "comparison_frames"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        for i in range(side_by_side.shape[0]):
            Image.fromarray(side_by_side[i]).save(comparison_dir / f"comparison_{i:04d}.png")
        images_to_video(side_by_side, output_path=str(comparison_video_path), fps=30, gradio_codec=False, verbose=True)
        logger.info(f"Saved RGB + FLAME mesh comparison video to {comparison_video_path}")
        
        # Add audio to comparison video
        if audio_source and os.path.exists(audio_source):
            add_audio_to_video(str(comparison_video_path), audio_source)


# -----------------------------------------------------------------------------
@torch.no_grad()
def infer_circle_around_video(cfg: DictConfig, fps: int = 30, seconds: int = 4, resolution: tuple = (512, 512)):
    """
    Renders a high-quality video circling around the head using the source pose.
    
    Args:
        cfg: Configuration object with dataset and experiment settings
        fps: Frames per second for the output video
        seconds: Duration of each trajectory segment in seconds
        resolution: Output resolution (H, W)
    """
    dev = torch.device("cuda")
    
    # ---------- Dataset ----------
    dataset = CafcaLamDataset(
        subject_list=list(cfg.dataset.cafca_subject_ids_val),
        num_source_frames=cfg.dataset.num_of_src_views,
        num_driving_frames=cfg.dataset.num_of_target_views,
        image_size=cfg.training.image_size,
        is_val=True,
        max_cache_size=cfg.dataset.get("max_cache_size", 128)
    )

    logger.info(f"Inference set: {len(dataset)} subjects")

    # ---------- Lightning model ----------
    finetuned_ckpt = cfg.experiment.checkpoint
    assert finetuned_ckpt and Path(finetuned_ckpt).exists(), "Checkpoint not found"
    lit = LamLightningModel.load_from_checkpoint(finetuned_ckpt)
    lit.eval().cuda()

    # Setup output directory
    out_root = Path(cfg.experiment.output_dir)
    video_name = "circle_around_videos_" + str(cfg.experiment.get("subj_id", 0))
    video_root = out_root / video_name
    video_root.mkdir(parents=True, exist_ok=True)

    # ---------- Get source data ----------
    subj_id = cfg.experiment.get("subj_id", 0)
    source_cam_ids = cfg.experiment.get("source_cam_ids", ["C0"])
    env_id = cfg.experiment.get("env_id", "002")
    expr_id = cfg.experiment.get("expr_id", "expr_00009")
    
    raw_batch = dataset.get_item_by_cam_ids(subj_id, source_cam_ids, env_id, expr_id)
    
    # Extract source data with correct keys from dataset output
    # Dataset returns tensors WITHOUT batch dimension, we need to add it
    source_intrs = raw_batch["source_intrs"].to(dev)           # [N_src, 4, 4]
    source_w2cs = raw_batch["source_w2cs"].to(dev)             # [N_src, 4, 4]
    source_tokens = raw_batch["tokens"].to(dev).float()        # [N_src, num_tokens, token_dim] - convert from float16
    source_canon_2_cam = raw_batch["source_canon_2_cam"].to(dev)  # [N_src, 4, 4]
    betas = raw_batch["betas"].to(dev)                         # [1, D] or [D]
    uid = raw_batch["uid"]
    
    # Add batch dimension - dataset returns WITHOUT batch dim
    # source_intrs: [N_src, 4, 4] -> [1, N_src, 4, 4]
    source_intrs = source_intrs.unsqueeze(0)
    # source_w2cs: [N_src, 4, 4] -> [1, N_src, 4, 4]  
    source_w2cs = source_w2cs.unsqueeze(0)
    # source_tokens: [N_src, num_tokens, token_dim] -> [1, N_src, num_tokens, token_dim]
    source_tokens = source_tokens.unsqueeze(0)
    # source_canon_2_cam: [N_src, 4, 4] -> [1, N_src, 4, 4]
    source_canon_2_cam = source_canon_2_cam.unsqueeze(0)
    # betas: [D] or [1, D] -> [1, D]
    if betas.dim() == 1:
        betas = betas.unsqueeze(0)

    H, W = resolution
    total_frames = seconds * fps

    # Generate circular trajectories around different axes
    # Trajectory 1: Circle around Z axis (looking from front, rotating left/right)
    trajectory1 = circle_around_axis(
        total_frames,
        axis=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        move=Vec3(0, 0, 1),
        distance=0.3,
    )
    
    # Trajectory 2: Circle around Y axis (horizontal orbit around head)
    # trajectory2 = circle_around_axis(
    #     total_frames,
    #     axis=Vec3(0, 1, 0),
    #     up=Vec3(0, 1, 0),
    #     move=Vec3(0, 0, -1),
    #     distance=0.3,
    # )
    
    # Trajectory 3: Circle from left side (X+)
    trajectory3 = circle_around_axis(
        total_frames,
        axis=Vec3(0, 0, 1),
        up=Vec3(0, 1, 0),
        move=Vec3(1, 0, 0),
        distance=0.3,
    )
    
    # Trajectory 4: Circle from right side (X-)
    trajectory4 = circle_around_axis(
        total_frames,
        axis=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        move=Vec3(-1, 0, 0),
        distance=0.3,
    )

    # Convert camera poses (c2w) to world-to-camera matrices (w2c)
    all_trajectories = trajectory1  + trajectory3 + trajectory4
    render_w2cs = torch.stack(
        [torch.from_numpy(np.linalg.inv(p)).float() for p in all_trajectories], dim=0
    ).unsqueeze(0).to(dev)  # [1, N_total, 4, 4]
    
    total_render_frames = len(all_trajectories)

    # Use first source view intrinsics for rendering, expand for all frames
    render_intr = source_intrs[:, 0:1, :, :]  # [1, 1, 4, 4]
    render_intrs = render_intr.expand(-1, total_render_frames, -1, -1)  # [1, N_total, 4, 4]

    # Build flame params for rendering (using source expression - neutral pose around head)
    # For circle-around we use a neutral/static expression
    N = total_render_frames
    flame_params = {
        "betas": betas,  # [1, D] - shape stays the same
        "expr": torch.zeros(1, N, 100, device=dev),
        "rotation": torch.zeros(1, N, 3, device=dev),
        "neck_pose": torch.zeros(1, N, 3, device=dev),
        "jaw_pose": torch.zeros(1, N, 3, device=dev),
        "eyes_pose": torch.zeros(1, N, 6, device=dev),
        "translation": torch.zeros(1, N, 3, device=dev),
    }

    # Background color (white)
    render_bg_colors = torch.ones((1, total_render_frames, 3), device=dev)

    # Prepare the batch for forward pass (using model's expected key names)
    render_batch = {
        "src_w2cs": source_w2cs,
        "src_intrs": source_intrs,
        "source_canon_2_cam": source_canon_2_cam,
        "render_w2cs": render_w2cs,
        "render_intrs": render_intrs,
        "render_bg_colors": render_bg_colors,
        "flame_params": flame_params,
        "latent_points": source_tokens,
    }

    video_path = video_root / f"{uid}_circle_around.mp4"
    
    logger.info(f"Rendering {total_render_frames} frames for circle-around video...")

    # Render frames in batches to manage memory
    chunk_size = cfg.experiment.get("render_chunk_size", 32)
    all_frames = []

    for start_idx in tqdm(range(0, total_render_frames, chunk_size), desc="Rendering chunks"):
        end_idx = min(start_idx + chunk_size, total_render_frames)
        
        # Create chunk batch
        chunk_batch = {
            "src_w2cs": render_batch["src_w2cs"],
            "src_intrs": render_batch["src_intrs"],
            "source_canon_2_cam": render_batch["source_canon_2_cam"],
            "render_w2cs": render_batch["render_w2cs"][:, start_idx:end_idx],
            "render_intrs": render_batch["render_intrs"][:, start_idx:end_idx],
            "render_bg_colors": render_batch["render_bg_colors"][:, start_idx:end_idx],
            "latent_points": render_batch["latent_points"],
            "flame_params": {
                k: v[:, start_idx:end_idx] if k != "betas" and v.dim() == 3 else v
                for k, v in render_batch["flame_params"].items()
            },
        }
        
        # Forward pass
        out = lit(chunk_batch)
        
        # Extract RGB frames [B, N_chunk, 3, H, W] -> [N_chunk, H, W, 3]
        rgb_chunk = out["comp_rgb"].detach().squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        rgb_chunk = (np.clip(rgb_chunk, 0, 1.0) * 255).astype(np.uint8)
        all_frames.append(rgb_chunk)

    # Concatenate all chunks
    all_frames = np.concatenate(all_frames, axis=0)

    # Save individual frames
    for i in range(all_frames.shape[0]):
        save_file = video_root / f"frame_{i:04d}.png"
        Image.fromarray(all_frames[i]).save(save_file)

    # Encode video with higher quality defaults (libx264, CRF-based)
    crf = cfg.experiment.get("circle_around_crf", 18)              # lower = higher quality
    bitrate = cfg.experiment.get("circle_around_bitrate", "8M")    # only used if CRF is not supported
    preset = cfg.experiment.get("circle_around_preset", "slow")    # slower = better compression/quality
    try:
        import imageio.v2 as iio
        with iio.get_writer(
            str(video_path),
            fps=fps,
            codec="libx264",
            format="FFMPEG",
            bitrate=bitrate,
            pixelformat="yuv420p",
            macro_block_size=None,  # keep odd resolutions valid
            output_params=["-crf", str(crf), "-preset", preset],
        ) as writer:
            for frame in all_frames:
                writer.append_data(frame)
        logger.info(f"Saved circle-around video to {video_path} (CRF={crf}, preset={preset})")
    except Exception as e:
        logger.warning(f"High-quality ffmpeg writer failed ({e}), falling back to mediapy.")
        with VideoWriter(str(video_path), (W, H), fps=fps) as writer:
            for i in range(all_frames.shape[0]):
                writer.add_image(all_frames[i])
        logger.info(f"Saved circle-around video to {video_path} via mediapy fallback")
    
    # Also save using images_to_video as backup
    backup_video_path = video_root / f"{uid}_circle_around_backup.mp4"
    images_to_video(all_frames, output_path=str(backup_video_path), fps=fps, gradio_codec=False, verbose=True)
    logger.info(f"Saved backup video to {backup_video_path}")

    return str(video_path)


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_infer_lam_cafca.py <config.yaml> "
              "[override_key=value ...] [experiment.mode=video|circle_around|all_pairs]")
        sys.exit(1)

    cfg = OmegaConf.load(sys.argv[1])
    overrides = OmegaConf.from_cli(sys.argv[2:])
    cfg = OmegaConf.merge(cfg, overrides)

    logger.info(OmegaConf.to_yaml(cfg))
    
    # Select inference mode based on config
    mode = cfg.experiment.get("mode", "video")
    
    if mode == "circle":
        # Render circle-around video using source pose
        fps = cfg.experiment.get("circle_around_fps", 30)
        seconds = cfg.experiment.get("circle_around_seconds", 4)
        resolution = (cfg.training.image_size, cfg.training.image_size)
        infer_circle_around_video(cfg, fps=fps, seconds=seconds, resolution=resolution)
    elif mode == "all_pairs":
        # Render all source-driving pairs
        infer_all_pairs(cfg, cfg.experiment.get("source_cam_ids", None), "env_000", "expr_00000")
    else:
        # Default: driven motion video
        infer_video(cfg)