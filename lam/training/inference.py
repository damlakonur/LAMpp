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

# -----------------------------------------------------------------------------
# add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root)) if str(project_root) not in sys.path else None
# -----------------------------------------------------------------------------
from lam.dataset.cafca_lam_dataset import CafcaLamDataset
from lam.training.lightning_lam_cafca import LamLightningModel
from lam.losses import LPIPSLoss, PixelLoss
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
)

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
    c2ws = motion_seqs["render_c2ws"].squeeze(0)         # [N, 4, 4]
    intrs = motion_seqs["render_intrs"].squeeze(0)       # [N, 3, 3]
    bg_colors = motion_seqs["render_bg_colors"].squeeze(0)  # [N, 3]

    # Replace driven views
    new_batch["render_w2cs"]     = torch.inverse(c2ws).unsqueeze(0).to(device)   # [1, N, 4, 4]
    new_batch["render_intrs"]    = intrs.unsqueeze(0).to(device)                 # [1, N, 3, 3] or [1, N, 4, 4]
    new_batch["render_bg_colors"] = bg_colors.unsqueeze(0).to(device)  

    # Override FLAME params (except betas)
    if "flame_params" in batch:
        for k, v in motion_seqs["flame_params"].items():
            if k == "betas":
                continue
            if k in batch["flame_params"]:
                if v.ndim == 2:
                    v = v.unsqueeze(0)  # [1, N, D]
                new_batch["flame_params"][k] = v.to(device)
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
    masks = (torch.stack([ds._load_image_as_tensor(f["mask_file_path"])
                          for f in frames])
             if want_mask else None)
    return rgbs, w2cs, intrs, masks
# -----------------------------------------------------------------------------
def build_source_dict(ds, subj_id, src_ids):
    """Return dict with the two source views (batch-dim already added)."""
    frames = [item for item in ds.subject_data[subj_id] if item["cam_id"] in src_ids]
    rgb, w2cs, intrs, _ = _stack_frame_list(ds, frames)
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
def infer_all_pairs(cfg, source_cam_ids=None):
    dev = torch.device("cuda")
    subj = int(cfg.experiment.subj_id)

    # Initialize dataset
    ds = CafcaLamDataset(
        [subj], 
        num_source_frames=2,
        num_driving_frames=0, 
        image_size=cfg.training.image_size,
        is_val=True, 
        mode="lam_infer"
    )
    if source_cam_ids is not None:
        # Map camera IDs to dataset indices
        cam_id_to_idx = {frame["cam_id"]: idx for idx, frame in enumerate(ds.subject_data[subj])}
        requested_indices = [cam_id_to_idx[cam_id] for cam_id in source_cam_ids]
        # Only use the requested indices if they're valid source candidates
        source_candidates = [idx for idx in requested_indices if idx in ds.source_candidates[subj]]
        if len(source_candidates) != len(source_cam_ids):
            raise ValueError(f"Not all requested camera IDs {source_cam_ids} are valid source candidates")
    else:
        source_candidates = ds.source_candidates[subj]
    
    all_frames = ds.subject_data[subj]
    
    # Load model
    ckpt = cfg.experiment.get("checkpoint", None)
    if ckpt and Path(ckpt).exists():
        lit = LamLightningModel.load_from_checkpoint(ckpt)
        logger.info(f"Loaded weights from {ckpt}")
    lit.eval().cuda()

    # Setup metrics
    psnr = PSNR(data_range=1.0).to(dev)
    ssim = SSIM(data_range=1.0).to(dev)
    lpips = LPIPSLoss(device=dev, prefetch=True)
    l1_fn = torch.nn.L1Loss()

    # Create output directory
    dir_name = "images" + "_" + str(cfg.experiment.get("subj_id", 0))
    out_root = Path(cfg.experiment.output_dir) / dir_name
    out_root.mkdir(parents=True, exist_ok=True)

    # Collect all source pair averages for summary
    all_pair_averages = []

    # For each possible pair of source candidate indices
    for i, j in combinations(source_candidates, 2):
        src_frames = [all_frames[i], all_frames[j]]
        src_ids = [f["cam_id"] for f in src_frames]
        
        # All other frames become driving frames
        drv_frames = [f for idx, f in enumerate(all_frames) if idx not in [i, j]]
        drv_ids = [f["cam_id"] for f in drv_frames]
        
        # Build source and driving dicts using helper functions
        src = build_source_dict(ds, subj, src_ids)
        drv = build_driving_dict(ds, subj, drv_ids)

        # Prepare model input
        batch = {
            "src_w2cs": src["source_w2cs"].to(dev),
            "src_intrs": src["source_intrs"].to(dev),
            "render_w2cs": drv["w2cs"].to(dev),
            "render_intrs": drv["intrs"].to(dev),
            "render_bg_colors": torch.ones((1, len(drv_ids), 3), device=dev),
            "flame_params": {k: v.to(dev) for k, v in drv["flame_params"].items()},
            "latent_points": src["tokens"].to(dev),
        }

        # Forward pass
        out = lit(batch)
        pred = out["comp_rgb"][0]  # [D,3,H,W]
        gt = drv["rgb"][0].to(dev)  # [D,3,H,W]

        # Initialize metrics list for this source pair
        pair_metrics = []

        # Save visualization for each driving view
        for k, drv_id in enumerate(drv_ids):
            # Create horizontal layout: [src1, src2, gt, pred]
            vis_tensors = [
                src["source_rgbs"][0, 0].cpu(),  
                src["source_rgbs"][0, 1].cpu(),  
                gt[k].cpu(),                     
                pred[k].cpu()                    
            ]
            
            # Concatenate horizontally and save
            vis = torch.cat(vis_tensors, dim=2)  # [3,H,W*4]
            vis_np = (vis.clamp(0, 1) * 255).byte().numpy()
            vis_np = np.transpose(vis_np, (1, 2, 0))  # [H,W*4,3]
            
            fn = out_root / f"subj{subj}_src{'-'.join(src_ids)}_drv{drv_id}.png"
            Image.fromarray(vis_np).save(fn)

            # Compute metrics
            l1 = l1_fn(pred[k], gt[k]).item()
            lp = lpips(pred[k:k+1], gt[k:k+1]).item()
            ps = psnr(pred[k:k+1], gt[k:k+1]).item()
            ss = ssim(pred[k:k+1], gt[k:k+1]).item()
            
            # Store metrics for averaging
            pair_metrics.append({
                'source_pair': '-'.join(src_ids),
                'driving_view': drv_id,
                'l1': l1,
                'lpips': lp,
                'psnr': ps,
                'ssim': ss
            })
            
            print(f"src {src_ids} → drv {drv_id} | "
                  f"L1 {l1:.3f} LPIPS {lp:.3f} PSNR {ps:.2f} SSIM {ss:.3f}")

        # # Compute and log averages for this source pair
        avg_l1 = np.mean([m['l1'] for m in pair_metrics])
        avg_lpips = np.mean([m['lpips'] for m in pair_metrics])
        avg_psnr = np.mean([m['psnr'] for m in pair_metrics])
        avg_ssim = np.mean([m['ssim'] for m in pair_metrics])
        
        # Store for overall summary
        all_pair_averages.append({
            'source_pair': '-'.join(src_ids),
            'avg_l1': avg_l1,
            'avg_lpips': avg_lpips,
            'avg_psnr': avg_psnr,
            'avg_ssim': avg_ssim
        })
        
        print(f"src {src_ids} AVERAGE | "
              f"L1 {avg_l1:.3f} LPIPS {avg_lpips:.3f} PSNR {avg_psnr:.2f} SSIM {avg_ssim:.3f}")

        # Write metrics to CSV file
        csv_file = out_root / f"subj{subj}_src{'-'.join(src_ids)}_metrics.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['source_pair', 'driving_view', 'l1', 'lpips', 'psnr', 'ssim'])
            for m in pair_metrics:
                writer.writerow([m['source_pair'], m['driving_view'], m['l1'], m['lpips'], m['psnr'], m['ssim']])
            # Add average row
            writer.writerow(['-'.join(src_ids), 'AVERAGE', avg_l1, avg_lpips, avg_psnr, avg_ssim])

        logger.info(f"Completed source pair {src_ids}")

    # Write overall summary CSV with all source pairs
    summary_csv = out_root / f"subj{subj}_all_pairs_summary.csv"
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source_pair', 'avg_l1', 'avg_lpips', 'avg_psnr', 'avg_ssim'])
        for avg in all_pair_averages:
            writer.writerow([avg['source_pair'], avg['avg_l1'], avg['avg_lpips'], avg['avg_psnr'], avg['avg_ssim']])
        
        # Compute overall averages across all source pairs
        overall_l1 = np.mean([avg['avg_l1'] for avg in all_pair_averages])
        overall_lpips = np.mean([avg['avg_lpips'] for avg in all_pair_averages])
        overall_psnr = np.mean([avg['avg_psnr'] for avg in all_pair_averages])
        overall_ssim = np.mean([avg['avg_ssim'] for avg in all_pair_averages])
        writer.writerow(['OVERALL_AVERAGE', overall_l1, overall_lpips, overall_psnr, overall_ssim])
    
    print(f"\nOVERALL AVERAGE | "
          f"L1 {overall_l1:.3f} LPIPS {overall_lpips:.3f} PSNR {overall_psnr:.2f} SSIM {overall_ssim:.3f}")
    logger.info(f"Summary saved to {summary_csv}")

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
        max_tokens_in_ram = cfg.dataset.get("max_tokens_in_ram", None)
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
        lit = LamLightningModel.load_from_checkpoint(ckpt)
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
        max_tokens_in_ram = cfg.dataset.get("max_tokens_in_ram", None)
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
        lit = LamLightningModel.load_from_checkpoint(ckpt)
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
    shape_param = betas[0].to("cuda")  # [num_shape_coeffs]

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
        max_tokens_in_ram = cfg.dataset.get("max_tokens_in_ram", None)
    )

    logger.info(f"Inference set: {len(dataset)} subjects")

    # ---------- Lightning model ----------
    # ckpt = cfg.experiment.get("checkpoint", None)
    # if ckpt and Path(ckpt).exists():
    #     lit = LamLightningModel.load_from_checkpoint(ckpt)
    #     logger.info(f"Loaded weights from {ckpt}")
    
    finetuned_ckpt = cfg.experiment.checkpoint
    assert finetuned_ckpt and Path(finetuned_ckpt).exists(), "Checkpoint not found"
    lit = LamLightningModel.load_from_checkpoint(finetuned_ckpt)
    lit.eval().cuda()

    out_root = Path(cfg.experiment.output_dir)
    video_name = "videos" + "_" + str(cfg.experiment.get("subj_id", 0))
    video_root = out_root / video_name
    video_root.mkdir(parents=True, exist_ok=True)

    # ---------- Manually specify source cam IDs ----------
    subj_id = cfg.experiment.get("subj_id", 0)
    source_cam_ids = cfg.experiment.get("source_cam_ids", ["C0"])
    raw_batch = dataset.get_item_by_cam_ids(subj_id, source_cam_ids)  # Only source info
    batch = lit.transfer_batch_to_device(raw_batch, device="cuda", dataloader_idx=0)

    betas = raw_batch["betas"]
    if betas.ndim == 3:
        betas = betas[:, 0]
    shape_param = betas[0].to("cuda")  # [num_shape_coeffs]

    # ---------- Motion sequence generation ----------
    dump_video_path = video_root / f"{batch['uid'][0]}_driven_motion_video.mp4"

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
        vis_motion         = False,
        shape_param        = shape_param,
    )
    motion_seqs = {k: v.to("cuda") if torch.is_tensor(v) else v for k, v in motion_seqs.items()}

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


# -----------------------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_infer_lam_cafca.py <config.yaml> "
              "[override_key=value ...]")
        sys.exit(1)

    cfg = OmegaConf.load(sys.argv[1])
    overrides = OmegaConf.from_cli(sys.argv[2:])
    cfg = OmegaConf.merge(cfg, overrides)

    logger.info(OmegaConf.to_yaml(cfg))
    # infer_all_pairs(cfg, cfg.experiment.get("source_cam_ids", None))
    infer_video(cfg)