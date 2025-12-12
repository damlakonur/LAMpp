"""
Inference script for NerSemble data using precomputed latent tokens.

Loads:
- Tokens from precomputed output (e.g., outputs/nersemble_tokens/240/tokens/)
- FLAME params, intrinsics, w2cs from source data folder (e.g., assets/nersemble_data_damla/240/)

Usage:
    python scripts/nersemble_inference.py \
        --data_dir assets/nersemble_data_damla/240 \
        --tokens_dir assests/nersemble_tokens/240 \
        --output_dir outputs/nersemble_renders/240 \
        --source_cams 222200047 222200038 \
        --frame_idx 0 \
        --mode circle \
        --fps 30 --seconds 4
"""

import os
import sys
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import torch
from PIL import Image
from glob import glob
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lam.training.lightning_lam_cafca import LamLightningModel
from lam.utils.video import images_to_video
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3


def load_nersemble_source_data(data_dir: str, source_cam_ids: list, frame_idx: int, render_size: int = 512):
    """
    Load source data (FLAME params, intrinsics, w2cs) from NerSemble data folder.
    
    Args:
        data_dir: Path to NerSemble subject folder
        source_cam_ids: List of camera IDs to use as source views
        frame_idx: Frame index for FLAME params (rotation, translation, etc.)
        render_size: Render resolution (512x512)
    
    Returns:
        dict with intrinsics, w2cs, flame_params
    """
    data_dir = Path(data_dir)
    
    # Get original image size from first source image
    first_img_path = data_dir / f"{source_cam_ids[0]}.jpg"
    if not first_img_path.exists():
        first_img_path = list(data_dir.glob("*.jpg"))[0]
    orig_img = Image.open(first_img_path)
    orig_w, orig_h = orig_img.size  # PIL gives (W, H)
    print(f"Original image size: {orig_w}x{orig_h}")
    
    # Load original intrinsics
    intrinsics_path = data_dir / "intrinrics.npy"
    if not intrinsics_path.exists():
        intrinsics_path = data_dir / "intrinsics.npy"
    intr_3x3 = np.load(str(intrinsics_path))
    print(f"Original intrinsics: fx={intr_3x3[0,0]:.1f}, fy={intr_3x3[1,1]:.1f}, cx={intr_3x3[0,2]:.1f}, cy={intr_3x3[1,2]:.1f}")
    
    # Scale intrinsics to match preprocessing (fit to render_size maintaining aspect ratio + pad)
    scale = render_size / max(orig_w, orig_h)
    new_w, new_h = int(orig_w * scale), int(orig_h * scale)
    pad_x = (render_size - new_w) // 2
    pad_y = (render_size - new_h) // 2
    
    print(f"Scaling: {orig_w}x{orig_h} -> {new_w}x{new_h} (padded to {render_size}x{render_size})")
    print(f"  Scale factor: {scale:.4f}, padding: ({pad_x}, {pad_y})")
    
    intrinsics = np.eye(4, dtype=np.float32)
    intrinsics[0, 0] = intr_3x3[0, 0] * scale  # fx
    intrinsics[1, 1] = intr_3x3[1, 1] * scale  # fy
    intrinsics[0, 2] = intr_3x3[0, 2] * scale + pad_x  # cx + padding offset
    intrinsics[1, 2] = intr_3x3[1, 2] * scale + pad_y  # cy + padding offset
    
    print(f"Scaled intrinsics: fx={intrinsics[0,0]:.1f}, fy={intrinsics[1,1]:.1f}, cx={intrinsics[0,2]:.1f}, cy={intrinsics[1,2]:.1f}")
    
    # Load w2c for each source camera
    w2cs = []
    for cam_id in source_cam_ids:
        w2c_path = data_dir / f"opencv_w2c_{cam_id}.npy"
        if w2c_path.exists():
            w2cs.append(np.load(str(w2c_path)))
        else:
            print(f"[WARN] W2C not found for camera {cam_id}, using identity")
            w2cs.append(np.eye(4, dtype=np.float32))
    w2cs = np.stack(w2cs, axis=0)  # [N_src, 4, 4]
    
    # Load FLAME params
    flame_path = data_dir / "flame_params.npz"
    flame_raw = np.load(str(flame_path))
    
    n_frames = flame_raw['expression'].shape[0]
    print(f"FLAME params: {n_frames} frames, using frame_idx={frame_idx}")
    
    # NerSemble keys -> Model expected keys:
    #   shape -> betas
    #   expression -> expr
    #   rotation -> rotation
    #   translation -> translation
    #   jaw -> jaw_pose
    #   neck -> neck_pose
    #   eyes -> eyes_pose
    
    # Get params for the specific frame
    flame_params = {
        "betas": torch.from_numpy(flame_raw["shape"]).float(),  # [1, 300]
        "expr": torch.from_numpy(flame_raw["expression"][frame_idx:frame_idx+1]).float(),  # [1, 100]
        "rotation": torch.from_numpy(flame_raw["rotation"][frame_idx:frame_idx+1]).float(),  # [1, 3]
        "translation": torch.from_numpy(flame_raw["translation"][frame_idx:frame_idx+1]).float(),  # [1, 3]
        "jaw_pose": torch.from_numpy(flame_raw["jaw"][frame_idx:frame_idx+1]).float(),  # [1, 3]
        "neck_pose": torch.from_numpy(flame_raw["neck"][frame_idx:frame_idx+1]).float(),  # [1, 3]
        "eyes_pose": torch.from_numpy(flame_raw["eyes"][frame_idx:frame_idx+1]).float(),  # [1, 6]
    }
    
    print(f"  betas: {flame_params['betas'].shape}")
    print(f"  expr: {flame_params['expr'].shape}")
    print(f"  rotation: {flame_params['rotation'].shape}")
    
    return {
        "intrinsics": intrinsics,  # [4, 4]
        "w2cs": w2cs,  # [N_src, 4, 4]
        "flame_params": flame_params,
    }


def load_tokens(tokens_dir: str, source_cam_ids: list):
    """
    Load precomputed tokens for source cameras.
    
    Args:
        tokens_dir: Directory containing tokens/*.npz
        source_cam_ids: List of camera IDs
    
    Returns:
        tokens: [N_src, N_pts, D] tensor
    """
    tokens_dir = Path(tokens_dir) / "tokens"
    
    tokens_list = []
    for cam_id in source_cam_ids:
        token_path = tokens_dir / f"{cam_id}.npz"
        if not token_path.exists():
            raise FileNotFoundError(f"Token file not found: {token_path}")
        token_data = np.load(token_path)
        tokens_list.append(torch.from_numpy(token_data["tokens"]).float())
        print(f"Loaded tokens for {cam_id}: shape {tokens_list[-1].shape}")
    
    tokens = torch.stack(tokens_list, dim=0)  # [N_src, N_pts, D]
    return tokens


def generate_circle_trajectory(fps: int, seconds: int, distance: float = 0.3):
    """
    Generate circular trajectories using dreifus (same as inference.py).
    
    Returns:
        render_w2cs: list of [4, 4] numpy arrays
    """
    total_frames = fps * seconds
    
    # Trajectory 1: Circle around Z axis
    trajectory1 = circle_around_axis(
        total_frames,
        axis=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        move=Vec3(0, 0, 1),
        distance=distance,
    )
    
    # Trajectory 3: Circle from left side (X+)
    trajectory3 = circle_around_axis(
        total_frames,
        axis=Vec3(0, 0, 1),
        up=Vec3(0, 1, 0),
        move=Vec3(1, 0, 0),
        distance=distance,
    )
    
    # Trajectory 4: Circle from right side (X-)
    trajectory4 = circle_around_axis(
        total_frames,
        axis=Vec3(0, 0, -1),
        up=Vec3(0, 1, 0),
        move=Vec3(-1, 0, 0),
        distance=distance,
    )
    
    # Combine trajectories (c2w matrices)
    all_trajectories = trajectory1 + trajectory3 + trajectory4
    
    # Convert c2w to w2c
    render_w2cs = [np.linalg.inv(p).astype(np.float32) for p in all_trajectories]
    
    return render_w2cs


@torch.no_grad()
def render_frames(
    model: LamLightningModel,
    tokens: torch.Tensor,
    source_w2cs: torch.Tensor,
    source_intrs: torch.Tensor,
    render_w2cs: torch.Tensor,
    render_intrs: torch.Tensor,
    flame_params: dict,
    device: torch.device,
    chunk_size: int = 32,
) -> np.ndarray:
    """
    Render frames using the model.
    """
    n_frames = render_w2cs.shape[1]
    all_frames = []
    
    # Identity source_canon_2_cam
    n_src = tokens.shape[1]
    source_canon_2_cam = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(1, n_src, -1, -1)
    
    for start_idx in tqdm(range(0, n_frames, chunk_size), desc="Rendering"):
        end_idx = min(start_idx + chunk_size, n_frames)
        chunk_size_actual = end_idx - start_idx
        
        # Slice render cameras and FLAME params for this chunk
        chunk_batch = {
            "src_w2cs": source_w2cs,
            "src_intrs": source_intrs,
            "source_canon_2_cam": source_canon_2_cam,
            "render_w2cs": render_w2cs[:, start_idx:end_idx],
            "render_intrs": render_intrs[:, start_idx:end_idx],
            "render_bg_colors": torch.ones((1, chunk_size_actual, 3), device=device),
            "latent_points": tokens,
            "flame_params": {
                k: v[:, start_idx:end_idx] if k != "betas" and v.dim() == 3 else v
                for k, v in flame_params.items()
            },
        }
        
        # Forward pass
        out = model(chunk_batch)
        
        # Extract RGB [1, N, 3, H, W] -> [N, H, W, 3]
        rgb = out["comp_rgb"].squeeze(0).permute(0, 2, 3, 1).cpu().numpy()
        rgb_uint8 = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        all_frames.append(rgb_uint8)
    
    return np.concatenate(all_frames, axis=0)


def main():
    parser = argparse.ArgumentParser(description="Render novel views from NerSemble data")
    parser.add_argument("--data_dir", type=str, required=True, help="NerSemble source data folder")
    parser.add_argument("--tokens_dir", type=str, required=True, help="Precomputed tokens folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--config", type=str, default="configs/inference/inference_lam_cafca.yaml", help="Config")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint (overrides config)")
    parser.add_argument("--source_cams", type=str, nargs="+", required=True, help="Source camera IDs")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index for FLAME params")
    parser.add_argument("--mode", type=str, choices=["circle", "cameras"], default="circle", help="Render mode")
    parser.add_argument("--render_cams", type=str, nargs="*", help="Camera IDs to render (for mode=cameras)")
    parser.add_argument("--fps", type=int, default=30, help="FPS for video")
    parser.add_argument("--seconds", type=int, default=4, help="Seconds per trajectory (mode=circle)")
    parser.add_argument("--distance", type=float, default=0.3, help="Orbit distance (mode=circle)")
    parser.add_argument("--chunk_size", type=int, default=32, help="Render chunk size")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    # Load model
    ckpt_path = args.checkpoint or cfg.experiment.checkpoint
    print(f"Loading model from {ckpt_path}...")
    model = LamLightningModel.load_from_checkpoint(ckpt_path)
    model.eval().to(device)
    
    # Load source data (FLAME, intrinsics, w2cs) from data_dir
    print(f"Loading source data from {args.data_dir}...")
    source_data = load_nersemble_source_data(args.data_dir, args.source_cams, args.frame_idx)
    
    # Load tokens from tokens_dir
    print(f"Loading tokens from {args.tokens_dir}...")
    tokens = load_tokens(args.tokens_dir, args.source_cams)
    tokens = tokens.unsqueeze(0).to(device)  # [1, N_src, N_pts, D]
    
    # Prepare source cameras
    source_w2cs = torch.from_numpy(source_data["w2cs"]).float().unsqueeze(0).to(device)  # [1, N_src, 4, 4]
    intr = torch.from_numpy(source_data["intrinsics"]).float().to(device)  # [4, 4]
    source_intrs = intr.unsqueeze(0).unsqueeze(0).expand(1, len(args.source_cams), -1, -1)  # [1, N_src, 4, 4]
    
    # Generate render cameras based on mode
    if args.mode == "circle":
        print(f"Generating circle trajectory: {args.fps} fps x {args.seconds} sec x 3 trajectories")
        render_w2cs_list = generate_circle_trajectory(args.fps, args.seconds, args.distance)
        render_w2cs = torch.stack([torch.from_numpy(w) for w in render_w2cs_list], dim=0)
        render_w2cs = render_w2cs.unsqueeze(0).to(device)  # [1, N, 4, 4]
    else:
        # Use specific cameras from the data
        if not args.render_cams:
            raise ValueError("--render_cams required for mode=cameras")
        render_w2cs_list = []
        for cid in args.render_cams:
            w2c_path = Path(args.data_dir) / f"opencv_w2c_{cid}.npy"
            if not w2c_path.exists():
                raise FileNotFoundError(f"W2C not found: {w2c_path}")
            render_w2cs_list.append(np.load(str(w2c_path)))
        render_w2cs = torch.from_numpy(np.stack(render_w2cs_list, axis=0)).float()
        render_w2cs = render_w2cs.unsqueeze(0).to(device)
    
    n_render = render_w2cs.shape[1]
    render_intrs = intr.unsqueeze(0).unsqueeze(0).expand(1, n_render, -1, -1).contiguous()
    
    # Prepare FLAME params - expand to N_render frames
    # For circle mode, use neutral expression (zeros)
    betas = source_data["flame_params"]["betas"].to(device)  # [1, 300]
    
    if args.mode == "circle":
        print("Using neutral expression for orbit rendering")
        flame_params = {
            "betas": betas,
            "expr": torch.zeros(1, n_render, 100, device=device),
            "rotation": torch.zeros(1, n_render, 3, device=device),
            "neck_pose": torch.zeros(1, n_render, 3, device=device),
            "jaw_pose": torch.zeros(1, n_render, 3, device=device),
            "eyes_pose": torch.zeros(1, n_render, 6, device=device),
            "translation": torch.zeros(1, n_render, 3, device=device),
        }
    else:
        # Use source frame's FLAME params, expanded to all render views
        flame_params = {
            "betas": betas,
            "expr": source_data["flame_params"]["expr"].unsqueeze(0).expand(1, n_render, -1).to(device),
            "rotation": source_data["flame_params"]["rotation"].unsqueeze(0).expand(1, n_render, -1).to(device),
            "neck_pose": source_data["flame_params"]["neck_pose"].unsqueeze(0).expand(1, n_render, -1).to(device),
            "jaw_pose": source_data["flame_params"]["jaw_pose"].unsqueeze(0).expand(1, n_render, -1).to(device),
            "eyes_pose": source_data["flame_params"]["eyes_pose"].unsqueeze(0).expand(1, n_render, -1).to(device),
            "translation": source_data["flame_params"]["translation"].unsqueeze(0).expand(1, n_render, -1).to(device),
        }
    
    print(f"\nRendering {n_render} frames...")
    print(f"  Source views: {len(args.source_cams)}")
    print(f"  Tokens shape: {tokens.shape}")
    print(f"  Betas shape: {betas.shape}")
    
    # Render
    frames = render_frames(
        model, tokens, source_w2cs, source_intrs,
        render_w2cs, render_intrs, flame_params,
        device, chunk_size=args.chunk_size
    )
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save frames
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frames_dir / f"{i:04d}.png")
    
    # Save video
    video_path = output_dir / f"render_{args.mode}.mp4"
    images_to_video(frames, str(video_path), fps=args.fps, gradio_codec=False, verbose=True)
    
    print(f"\nSaved {len(frames)} frames to {frames_dir}")
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()

