# run_infer_lam_cafca.py
import os, sys, csv, random, logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf, DictConfig
import pytorch_lightning as pl
from torchvision.utils import make_grid
from PIL import Image
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
            src_rgb = batch["image"].clamp(0, 1)                # [Ns, 3, H, W]
            gt_rgb = batch["gt_render_images"].clamp(0, 1)      # [Nv, 3, H, W]
            pred_rgb = rgb[0].clamp(0, 1)                       # [Nv, 3, H, W]

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
    ckpt = cfg.experiment.get("checkpoint", None)
    if ckpt and Path(ckpt).exists():
        lit = LamLightningModel.load_from_checkpoint(ckpt)
        logger.info(f"Loaded weights from {ckpt}")

    lit.eval().cuda()

    out_root = Path(cfg.experiment.output_dir)
    (out_root / "images").mkdir(parents=True, exist_ok=True)
    video_root = out_root / "videos"
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
    infer_video(cfg)