from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from loguru import logger
from omegaconf import OmegaConf
from safetensors.torch import load_file

from lam.models.modeling_lam import ModelLAM
from lam.training.lightning_lam_cafca import LamLightningModel
from lam.runners.infer.head_utils import preprocess_image, prepare_motion_seqs
from lam.utils.video import images_to_video
from omegaconf import DictConfig

def build_model(cfg: DictConfig):
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
    
    return model

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


def _compute_latent_tokens(
    model: ModelLAM,
    rgb_tensor: torch.Tensor,              # [B, C, H, W]  (single view)
    flame_params: dict,
) -> torch.Tensor:
    """Runs the encoder/transformer on ONE image and returns latent tokens.

    Returns tensor of shape [B, N_pts, D]."""

    qp = None
    if model.latent_query_points_type.startswith("e2e_flame"):
        qp, _, _, _ = model.renderer.get_query_points(flame_params, device=rgb_tensor.device)

    tokens, image_feats = model.forward_latent_points(
        image=rgb_tensor, camera=None, query_points=qp
    )  # [B, N_pts, D]
    return tokens


# -------------------------------------------------------------- #
# PUBLIC FUNCTIONS
# -------------------------------------------------------------- #


def save_latent_points(
    cfg_path: str,
    ref_images: List[str],
    latent_out_path: str,
    tmp_dir: str = "./tmp",
):
    """Extracts latent tokens for exactly two reference images and saves them.

    The function performs flame-tracking, image pre-processing and uses a *full*
    LAM (encoder + transformer) built from the base checkpoint to compute the
    latent point tokens.  The result is written to *latent_out_path* as a
    compressed npz containing:

    • latent   – [1, 2, N_pts, D] float16
    • shape_param
    • src_intrs
    """
    
    # ------------------------------------------------------------
    # 0. Load config & build model
    # ------------------------------------------------------------
    cfg = OmegaConf.load(cfg_path)
    model: ModelLAM = build_model(cfg)
    model.eval()
    model.cuda()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------
    # 1. Run FLAME tracking to obtain canonical images + params
    # ------------------------------------------------------------
    from tools.flame_tracking_single_image import FlameTrackingSingleImage

    flametracker = FlameTrackingSingleImage(
        output_dir=tmp_dir,
        alignment_model_path="./model_zoo/flame_tracking_models/68_keypoints_model.pkl",
        vgghead_model_path="./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd",
        human_matting_path="./model_zoo/flame_tracking_models/matting/stylematte_synth.pt",
        facebox_model_path="./model_zoo/flame_tracking_models/FaceBoxesV2.pth",
        detect_iris_landmarks=True,
        args=cfg,
    )

    intr_list: list[torch.Tensor] = []      # [3,3] each
    tokens_list: list[torch.Tensor] = []    # [1,N,D] each
    shape_param = None

    for img_path in ref_images:
        # run tracking
        _, mask_dir = flametracker.preprocess(img_path)
        flametracker.optimize()
        _, export_dir = flametracker.export()
        img_proc_path = os.path.join(export_dir, "images/00000_00.png")
        # Use the *exported* foreground mask that is spatially aligned with
        # the processed RGB (same 1024×1024 canvas) as done in training.
        mask_proc_path = img_proc_path.replace("/images/", "/fg_masks/")
        # load shape parameters (identical for the same identity)
        flame_params_path = os.path.join(export_dir, "canonical_flame_param.npz")
        flame_npz = np.load(flame_params_path)
        shape_param = torch.from_numpy(flame_npz["shape"]).float()

        # --------------------------------------------------------
        # 2. Pre-process image for the encoder
        # --------------------------------------------------------
        rgb, _, _, intr, _ = preprocess_image(
            img_proc_path,
            mask_path=mask_proc_path,
            intr=None,
            pad_ratio=0,
            bg_color=1.0,
            max_tgt_size=None,
            aspect_standard=1.0,
            enlarge_ratio=[1.0, 1.0],
            render_tgt_size=512,
            multiply=14,
            need_mask=False,
            get_shape_param=False,
        )
        # -------- latent token for this single view --------
        with torch.no_grad():
            flame_param_dict = {"betas": shape_param.unsqueeze(0).to(device)}
            # Ensure rgb has batch dimension [B, C, H, W]
            rgb_input = rgb.to(device)
            if rgb_input.dim() == 3:
                rgb_input = rgb_input.unsqueeze(0)  # [C, H, W] -> [1, C, H, W]
            tok_single = _compute_latent_tokens(model, rgb_input, flame_param_dict)  # [1,N,D]
            tokens_list.append(tok_single)


        # prepare intrinsics
        # Keep intrinsics in 3×3 form; fallback to identity if not returned.
        if intr is None:
            intr = torch.eye(3)
        elif intr.shape[-1] == 4:   # sometimes 4×4 => take the 3×3 top-left
            intr = intr[:3, :3]
        intr_list.append(intr.float())

    # ------------------------------------------------------------
    # 3. Stack per-view tokens -> final latent tensor [1,2,N,D]
    # ------------------------------------------------------------
    latent_tokens = torch.stack(tokens_list, dim=1)  # [1,2,N,D]

    # ------------------------------------------------------------
    # 4. Save to disk
    # ------------------------------------------------------------
    latent_np = latent_tokens.half().cpu().numpy()
    src_intrs_np = torch.stack(intr_list, dim=0).cpu().numpy()       # [2,3,3]

    Path(os.path.dirname(latent_out_path)).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(latent_out_path,
                        latent=latent_np,
                        shape_param=shape_param.cpu().numpy(),
                        src_intrs=src_intrs_np)

    logger.success(f"Saved latent tokens to {latent_out_path}")


# -------------------------------------------------------------- #
#  NEW: batch-style inference like training/inference.py but using
#        pre-computed latent tokens instead of images.
# -------------------------------------------------------------- #


@torch.no_grad()
def infer_video_with_latents(
    cfg_path: str,
    latent_npz: str,
    motion_seqs_dir: str,
    output_dir: str,
    fps: int = 30,
):
    """Run full video inference using latent tokens.

    Creates a proper model batch identical to training/inference.py and calls
    the LightningModule (lit) directly so that it executes the same forward
    path that was used during fine-tuning.
    """

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- load latents & metadata --------
    pack = np.load(latent_npz)
    latent = torch.from_numpy(pack["latent"]).float()           # [1,2,N,D]
    shape_param = torch.from_numpy(pack["shape_param"]).float() # [D_shape]
    src_intrs_np = pack["src_intrs"]                             # [2,3,3]
    src_intrs = torch.from_numpy(src_intrs_np).float()
    src_w2cs = torch.eye(4).unsqueeze(0).repeat(2, 1, 1)        # identity
    src_canon_2_cam = torch.eye(4).unsqueeze(0).repeat(2, 1, 1) # identity
    # -------- cfg & model --------
    cfg = OmegaConf.load(cfg_path)
    cfg.model["instantiate_encoder"] = False
    cfg.model["instantiate_transformer"] = False

    ckpt = cfg.experiment.checkpoint
    assert ckpt and Path(ckpt).exists(), "Checkpoint not found"

    lit = LamLightningModel.load_from_checkpoint(ckpt)
    # lit = load_checkpoint_with_shape_mismatch_handling(ckpt, cfg)
    # breakpoint()
    
    lit.eval().cuda()

    # -------- prepare motion sequence --------
    motion_seq = prepare_motion_seqs(
        motion_seqs_dir,
        "",
        save_root=output_dir,
        fps=6,
        bg_color=1.0,
        aspect_standard=1.0,
        enlarge_ratio=[1.0, 1.0],
        render_image_res=512,
        multiply=16,
        need_mask=False,
        vis_motion=False,
        shape_param=shape_param,
    )
    motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)

    # -------- build batch --------
    batch = {
        "src_w2cs": src_w2cs.unsqueeze(0).to(dev),       # [1,2,4,4]
        "src_intrs": src_intrs.unsqueeze(0).to(dev),    # [1,2,3,3]
        "render_w2cs": torch.inverse(motion_seq["render_c2ws"]).to(dev),
        "render_intrs": motion_seq["render_intrs"].to(dev),
        "render_bg_colors": motion_seq["render_bg_colors"].to(dev),
        "flame_params": {k: v.to(dev) for k, v in motion_seq["flame_params"].items()},
        "latent_points": latent.to(dev),
        "source_canon_2_cam": src_canon_2_cam.to(dev),
    }

    # -------- forward --------
    out = lit(batch)
    rgb = out["comp_rgb"].detach().squeeze(0).permute(0, 2, 3, 1).cpu().numpy()  # [Nv,H,W,3]
    rgb_uint8 = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    from PIL import Image
    for i in range(rgb_uint8.shape[0]):
        save_file = os.path.join(output_dir, f"{i:04d}.png")
        Image.fromarray(rgb_uint8[i]).save(save_file)
    video_path = Path(output_dir) / "driven_video3.mp4"
    images_to_video(rgb_uint8, output_path=str(video_path), fps=fps, gradio_codec=False, verbose=True)
    logger.success(f"Saved driven video to {video_path}")


# -------------------------------------------------------------- #
# CLI
# -------------------------------------------------------------- #


def cli():
    p = argparse.ArgumentParser(description="Two-stage inference helper for 2-view LAM")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Command: extract
    p_ext = sub.add_parser("extract", help="Compute and save latent tokens")
    p_ext.add_argument("cfg", type=str, help="Config YAML")
    p_ext.add_argument("img0", type=str, help="Reference image 1")
    p_ext.add_argument("img1", type=str, help="Reference image 2")
    p_ext.add_argument("out", type=str, help="Path to save .npz with latents")
    p_ext.add_argument("--tmp", type=str, default="./tmp", help="Temp dir for tracking")

    # Command: video
    p_vid = sub.add_parser("video", help="Run full video inference via Lightning batch API")
    p_vid.add_argument("cfg", type=str, help="Config YAML")
    p_vid.add_argument("latents", type=str, help="Latent .npz file")
    p_vid.add_argument("motion_seqs", type=str, help="Motion seq directory")
    p_vid.add_argument("out_dir", type=str, help="Output directory for video & frames")
    p_vid.add_argument("--fps", type=int, default=30, help="FPS for video")

    args = p.parse_args()

    if args.cmd == "extract":
        save_latent_points(
            cfg_path=args.cfg,
            ref_images=[args.img0, args.img1],
            latent_out_path=args.out,
            tmp_dir=args.tmp,
        )
    elif args.cmd == "video":
        infer_video_with_latents(
            cfg_path=args.cfg,
            latent_npz=args.latents,
            motion_seqs_dir=args.motion_seqs,
            output_dir=args.out_dir,
            fps=args.fps,
        )


if __name__ == "__main__":
    cli()