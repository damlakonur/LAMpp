"""
Precompute latent tokens for NerSemble data.

NerSemble data structure (per subject folder):
    - *.jpg: Camera images (e.g., 222200047.jpg)
    - intrinrics.npy: [3, 3] camera intrinsics (shared across cameras)
    - opencv_w2c_CAMERA_ID.npy: [4, 4] world-to-camera for each camera
    - flame_params.npz: FLAME parameters with keys:
        - shape: [1, 300]
        - expression: [N_frames, 100]
        - rotation: [N_frames, 3]
        - translation: [N_frames, 3]
        - jaw: [N_frames, 3]
        - neck: [N_frames, 3]
        - eyes: [N_frames, 6]
        - frames: [N_frames] frame indices

Usage:
    python scripts/nersemble_precompute.py --data_dir assets/nersemble_data_damla/240 --output_dir outputs/nersemble_tokens/240
"""

import os
import sys
import argparse
from pathlib import Path
from glob import glob

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from safetensors.torch import load_file
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from lam.models.modeling_lam import ModelLAM
from external.human_matting import StyleMatteEngine


def build_model(cfg):
    """Build and load the LAM model."""
    model = ModelLAM(**cfg.model)
    resume = os.path.join(cfg.experiment.model_name, "model.safetensors")
    print("=" * 60)
    print(f"Loading pretrained weights from: {resume}")
    
    if resume.endswith('safetensors'):
        ckpt = load_file(resume, device='cpu')
    else:
        ckpt = torch.load(resume, map_location='cpu')
    
    state_dict = model.state_dict()
    loaded, skipped = 0, 0
    for k, v in ckpt.items():
        if k in state_dict:
            if state_dict[k].shape == v.shape:
                state_dict[k].copy_(v)
                loaded += 1
            else:
                print(f"[WARN] Shape mismatch for {k}: ckpt {v.shape} vs model {state_dict[k].shape}")
                skipped += 1
        else:
            skipped += 1
    
    print(f"Loaded {loaded} params, skipped {skipped}")
    print("=" * 60)
    return model


def load_nersemble_data(data_dir: str):
    """
    Load NerSemble data from a subject directory.
    
    Returns:
        dict with:
            - images: dict of {cam_id: PIL.Image}
            - intrinsics: [3, 3] numpy array
            - w2cs: dict of {cam_id: [4, 4] numpy array}
            - flame_params: dict of FLAME parameters
            - cam_ids: list of camera IDs
    """
    data_dir = Path(data_dir)
    
    # Find all camera images
    image_files = sorted(glob(str(data_dir / "*.jpg")))
    cam_ids = [Path(f).stem for f in image_files]
    
    print(f"Found {len(cam_ids)} cameras: {cam_ids}")
    
    # Load images
    images = {}
    for cam_id, img_path in zip(cam_ids, image_files):
        images[cam_id] = Image.open(img_path).convert("RGB")
    
    # Load intrinsics (shared)
    intrinsics_path = data_dir / "intrinrics.npy"
    if not intrinsics_path.exists():
        intrinsics_path = data_dir / "intrinsics.npy"  # try alternative spelling
    intrinsics = np.load(str(intrinsics_path))  # [3, 3]
    print(f"Intrinsics shape: {intrinsics.shape}")
    
    # Load w2c for each camera
    w2cs = {}
    for cam_id in cam_ids:
        w2c_path = data_dir / f"opencv_w2c_{cam_id}.npy"
        if w2c_path.exists():
            w2cs[cam_id] = np.load(str(w2c_path))  # [4, 4]
        else:
            print(f"[WARN] W2C not found for camera {cam_id}, using identity")
            w2cs[cam_id] = np.eye(4, dtype=np.float32)
    
    # Load FLAME params
    flame_path = data_dir / "flame_params.npz"
    flame_raw = np.load(str(flame_path))
    
    # Convert to model-expected format
    flame_params = {
        "betas": torch.from_numpy(flame_raw["shape"]).float(),  # [1, 300]
        "expr": torch.from_numpy(flame_raw["expression"]).float(),  # [N, 100]
        "rotation": torch.from_numpy(flame_raw["rotation"]).float(),  # [N, 3]
        "translation": torch.from_numpy(flame_raw["translation"]).float(),  # [N, 3]
        "jaw_pose": torch.from_numpy(flame_raw["jaw"]).float(),  # [N, 3]
        "neck_pose": torch.from_numpy(flame_raw["neck"]).float(),  # [N, 3]
        "eyes_pose": torch.from_numpy(flame_raw["eyes"]).float(),  # [N, 6]
        "frames": flame_raw["frames"],  # frame indices
    }
    
    print(f"FLAME params: betas {flame_params['betas'].shape}, {flame_params['expr'].shape[0]} frames")
    
    return {
        "images": images,
        "intrinsics": intrinsics,
        "w2cs": w2cs,
        "flame_params": flame_params,
        "cam_ids": cam_ids,
    }


def preprocess_image(
    img: Image.Image, 
    target_size: int = 512,
    matting_engine: StyleMatteEngine = None,
    bg_color: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Preprocess image for the encoder with optional background masking.
    
    Args:
        img: PIL Image
        target_size: Target size for saving (512x512)
        matting_engine: StyleMatteEngine for background removal (optional)
        bg_color: Background color (0=black, 1=white)
    
    Returns:
        Tuple of:
            - img_512: [C, 512, 512] for saving
            - img_504: [C, 504, 504] for encoder (DINOv2 compatible)
            - mask: [512, 512] alpha mask (or None if no matting)
    """
    w, h = img.size
    
    # Resize to fit within 512x512 (maintain aspect ratio)
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = img.resize((new_w, new_h), Image.LANCZOS)
    
    # Pad to 512x512 square (center the image, fill with bg_color)
    bg_value = int(bg_color * 255)
    img_512 = Image.new("RGB", (target_size, target_size), (bg_value, bg_value, bg_value))
    paste_x = (target_size - new_w) // 2
    paste_y = (target_size - new_h) // 2
    img_512.paste(img_resized, (paste_x, paste_y))
    
    # Convert to tensor [C, H, W] in [0, 1]
    img_tensor_512 = T.ToTensor()(img_512)
    
    # Apply background masking if matting engine provided
    mask = None
    if matting_engine is not None:
        with torch.no_grad():
            img_tensor_device = img_tensor_512.to(matting_engine._device)
            matted_img, alpha = matting_engine(img_tensor_device, return_type='matting', background_rgb=bg_color)
            img_tensor_512 = matted_img.cpu()
            mask = alpha.cpu()
    
    # Create 504x504 version for encoder (DINOv2 patch size = 14, 504 = 14*36)
    img_tensor_504 = torch.nn.functional.interpolate(
        img_tensor_512.unsqueeze(0), size=(504, 504), mode='bilinear', align_corners=False
    ).squeeze(0)
    
    return img_tensor_512, img_tensor_504, mask


def compute_latent_tokens(
    model: ModelLAM,
    image: torch.Tensor,
    flame_params: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute latent tokens for a single image.
    
    Args:
        model: LAM model
        image: [C, H, W] tensor
        flame_params: dict with betas (for query points)
        device: torch device
    
    Returns:
        tokens: [N_pts, D] tensor
    """
    # Add batch dimension
    image = image.unsqueeze(0).to(device)  # [1, C, H, W]
    
    # Get query points from FLAME
    query_points = None
    if model.latent_query_points_type.startswith("e2e_flame"):
        # Need betas for query points
        flame_for_qp = {"betas": flame_params["betas"].to(device)}
        query_points, _, _, _ = model.renderer.get_query_points(flame_for_qp, device=device)
    
    # Forward through encoder + transformer
    tokens, image_feats = model.forward_latent_points(
        image, camera=None, query_points=query_points, additional_features={}
    )
    
    return tokens.squeeze(0)  # [N_pts, D]


def main():
    parser = argparse.ArgumentParser(description="Precompute latent tokens for NerSemble data")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to NerSemble subject folder")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tokens")
    parser.add_argument("--config", type=str, default="configs/inference/inference_lam_cafca.yaml", help="Config file")
    parser.add_argument("--frame_idx", type=int, default=0, help="Frame index for FLAME params (default: 0)")
    parser.add_argument("--image_size", type=int, default=512, help="Image size for encoder")
    parser.add_argument("--no_matting", action="store_true", help="Skip background matting")
    parser.add_argument("--matting_path", type=str, 
                        default="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
                        help="Path to matting model weights")
    parser.add_argument("--bg_color", type=float, default=1.0, help="Background color (0=black, 1=white)")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load config
    cfg = OmegaConf.load(args.config)
    
    # Build model
    print("Building model...")
    model = build_model(cfg)
    model.to(device)
    model.eval()
    
    # Initialize matting engine
    matting_engine = None
    if not args.no_matting:
        if os.path.exists(args.matting_path):
            print(f"Loading matting model from {args.matting_path}...")
            matting_engine = StyleMatteEngine(device=str(device), human_matting_path=args.matting_path)
            print("Matting engine ready - will remove backgrounds")
        else:
            print(f"[WARN] Matting model not found at {args.matting_path}, skipping background removal")
    else:
        print("Background matting disabled (--no_matting)")
    
    # Load NerSemble data
    print(f"Loading data from {args.data_dir}...")
    data = load_nersemble_data(args.data_dir)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Also save masked images for debugging
    masked_dir = output_dir / "masked_images"
    masked_dir.mkdir(exist_ok=True)
    
    # Compute tokens for each camera
    all_tokens = {}
    all_w2cs = {}
    all_masks = {}
    
    with torch.no_grad():
        for cam_id in tqdm(data["cam_ids"], desc="Computing tokens"):
            # Preprocess image with optional matting
            img = data["images"][cam_id]
            img_512, img_504, mask = preprocess_image(
                img, 
                target_size=512,
                matting_engine=matting_engine,
                bg_color=args.bg_color,
            )
            
            # Save 512x512 masked image for debugging
            masked_img = (img_512.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            Image.fromarray(masked_img).save(masked_dir / f"{cam_id}.png")
            
            # Save mask if available
            if mask is not None:
                mask_img = (mask.numpy() * 255).astype(np.uint8)
                Image.fromarray(mask_img).save(masked_dir / f"{cam_id}_mask.png")
                all_masks[cam_id] = mask.numpy()
            
            # Compute tokens using 504x504 image (DINOv2 compatible)
            tokens = compute_latent_tokens(
                model, img_504, data["flame_params"], device
            )
            
            all_tokens[cam_id] = tokens.cpu().numpy().astype(np.float16)
            all_w2cs[cam_id] = data["w2cs"][cam_id]
            
            print(f"  {cam_id}: tokens shape {tokens.shape}")
    
    # Save results
    # Save individual tokens per camera
    tokens_dir = output_dir / "tokens"
    tokens_dir.mkdir(exist_ok=True)
    
    for cam_id, tokens in all_tokens.items():
        np.savez_compressed(tokens_dir / f"{cam_id}.npz", tokens=tokens)
    
    # Save combined metadata
    # Pad intrinsics to 4x4 (keep original values - don't scale)
    intr_3x3 = data["intrinsics"]
    intr_4x4 = np.eye(4, dtype=np.float32)
    intr_4x4[:3, :3] = intr_3x3
    
    print(f"Original intrinsics (unchanged):")
    print(f"  fx={intr_4x4[0,0]:.1f}, fy={intr_4x4[1,1]:.1f}, cx={intr_4x4[0,2]:.1f}, cy={intr_4x4[1,2]:.1f}")
    
    metadata = {
        "cam_ids": data["cam_ids"],
        "intrinsics": intr_4x4,
        "w2cs": np.stack([all_w2cs[cid] for cid in data["cam_ids"]], axis=0),
        "betas": data["flame_params"]["betas"].numpy(),
        "frame_idx": args.frame_idx,
        "image_size": 512,  # Saved image size
    }
    
    # Save FLAME params for the selected frame
    flame_frame = {
        "betas": data["flame_params"]["betas"].numpy(),
        "expr": data["flame_params"]["expr"][args.frame_idx:args.frame_idx+1].numpy(),
        "rotation": data["flame_params"]["rotation"][args.frame_idx:args.frame_idx+1].numpy(),
        "translation": data["flame_params"]["translation"][args.frame_idx:args.frame_idx+1].numpy(),
        "jaw_pose": data["flame_params"]["jaw_pose"][args.frame_idx:args.frame_idx+1].numpy(),
        "neck_pose": data["flame_params"]["neck_pose"][args.frame_idx:args.frame_idx+1].numpy(),
        "eyes_pose": data["flame_params"]["eyes_pose"][args.frame_idx:args.frame_idx+1].numpy(),
    }
    
    np.savez_compressed(output_dir / "metadata.npz", **metadata, **{f"flame_{k}": v for k, v in flame_frame.items()})
    
    print(f"\nSaved tokens to {tokens_dir}")
    print(f"Saved metadata to {output_dir / 'metadata.npz'}")
    print(f"Camera IDs: {data['cam_ids']}")


if __name__ == "__main__":
    main()

