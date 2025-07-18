
import csv
import torch
import os
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from accelerate.logging import get_logger
from pathlib import Path


from lam.runners.infer.head_utils import preprocess_image, load_flame_params


from .base_inferrer import Inferrer
from lam.runners import REGISTRY_RUNNERS
from safetensors.torch import load_file
from lam.dataset import env_paths
from lam.dataset.cafca_dataset import CafcaDataset
from torchmetrics.image import (
    PeakSignalNoiseRatio as PSNR,
    StructuralSimilarityIndexMeasure as SSIM,
)
from lam.losses import LPIPSLoss, PixelLoss


logger = get_logger(__name__)


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--infer', type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    if args.config is not None:
        cfg = OmegaConf.load(args.config)
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split('_')[-1]
        )
        cfg.save_tmp_dump = os.path.join("exps", 'save_tmp', _relative_path)
        cfg.image_dump    = os.path.join("exps", 'images', _relative_path)
        cfg.video_dump    = os.path.join("exps", 'videos', _relative_path)
        cfg.mesh_dump     = os.path.join("exps", 'meshes', _relative_path)

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("save_tmp_dump", os.path.join("exps", cli_cfg.model_name, 'save_tmp'))
        cfg.setdefault("image_dump",    os.path.join("exps", cli_cfg.model_name, 'images'))
        cfg.setdefault("video_dump",    os.path.join("dumps", cli_cfg.model_name, 'videos'))
        cfg.setdefault("mesh_dump",     os.path.join("dumps", cli_cfg.model_name, 'meshes'))

    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)
    cfg.setdefault("save_img", True) 
    cfg.setdefault('use_cafca_dataset', True)
    cfg.setdefault('cafca_subject_id_for_single_infer', None)
    cfg.setdefault('cafca_camera_id_for_single_infer', None)
    cfg.setdefault('cafca_driving_camera_id_for_single', None)
    cfg.setdefault('logger', 'INFO')

    assert cfg.model_name is not None, "model_name is required"
    if not cfg.get('use_cafca_dataset', False):
        assert cfg.image_input is not None, "image_input is required"
        assert cfg.export_video or cfg.export_mesh, \
            "At least one of export_video or export_mesh should be True"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg



@REGISTRY_RUNNERS.register('infer.baseline')
class LAMInferrer(Inferrer):

    EXP_TYPE: str = 'baseline'

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
        eval_subjects = [32] #, 31, 32, 33, 34, 35, 36, 41, 45, 50, 51, 55
        self.l1_loss_fn    = torch.nn.L1Loss()
        self.lpips_loss_fn = LPIPSLoss(device='cuda', prefetch=True)
        self.psnr_metric   = PSNR(data_range=1.0).to('cuda')
        self.ssim_metric   = SSIM(data_range=1.0).to('cuda')

        self.cafca_loader = None
        if self.cfg.get('use_cafca_dataset', False):
            logger.info("Initializing CafcaLamDataset for LAM inference.")
            if not hasattr(env_paths, 'subjects_train') or not env_paths.subjects_train:
                raise ValueError("env_paths.subjects_train is not defined or is empty. Please set it for CafcaLamDataset.")
            subject_id = self.cfg.get('cafca_subject_id_for_single_infer', None)
            self.cafca_dataset = CafcaDataset(subject_list=eval_subjects, mode="lam_infer")
            self.cafca_loader = torch.utils.data.DataLoader(
                self.cafca_dataset, batch_size=1, shuffle=False, num_workers=2  # Batch size 1 for inference
            )

    def _build_model(self, cfg):
        """
        from lam.models import model_dict
        hf_model_cls = wrap_model_hub(model_dict[self.EXP_TYPE])
        model = hf_model_cls.from_pretrained(cfg.model_name)
        """
        from lam.models import ModelLAM
        model = ModelLAM(**cfg.model)

        resume = os.path.join(cfg.model_name, "model.safetensors")
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

    
    def infer_single(
            self,
            image_path: str,
            mask_path_for_preprocess: str,
            target_intrinsics: np.ndarray,   # [N,4,4]
            canonical_flame_path_for_subject: str,
            world2cam: np.ndarray,           # [N,4,4]
            dump_image_dir: str,
            dump_video_path: str,
            driving_image_path: list,        # list[str] length N
            driving_cam_ids: list,           # list[str] length N
            export_video: bool):

        # ---------- preprocess source ----------
        source_size = self.cfg.source_size
        eff_mask = mask_path_for_preprocess if os.path.exists(mask_path_for_preprocess) else None
        image, _, _, _, shape_param = preprocess_image(
            image_path, mask_path=eff_mask,
            intr=None, pad_ratio=0, bg_color=1.0,
            max_tgt_size=None, aspect_standard=1.0,
            enlarge_ratio=[1.0,1.0], render_tgt_size=source_size,
            multiply=14, need_mask=True, get_shape_param=True,
            canonical_flame_path_override=canonical_flame_path_for_subject)

        # ---------- stack driving cameras ----------
        # world2cam & intrinsics come in as [N,4,4]
        N = world2cam.shape[0]
        render_w2c  = torch.from_numpy(world2cam).float().unsqueeze(0)   # [1,N,4,4]
        render_intr = torch.from_numpy(target_intrinsics).float().unsqueeze(0)  # [1,N,4,4]
        bg_colors   = torch.ones((1, N, 3), dtype=torch.float32)

        # ---------- flame params ----------
        flame_params = load_flame_params(canonical_flame_path_for_subject)
        flame_params['betas'] = shape_param.unsqueeze(0)

        for k in ['expr','rotation','neck_pose','jaw_pose','eyes_pose','translation']:
            if k in flame_params:
                # make shape [1, 1, D]  →  [1, N, D]
                flame_params[k] = flame_params[k].unsqueeze(0).unsqueeze(0).repeat(1, N, 1)

        # ---------- inference ----------
        device, dtype = 'cuda', torch.float32
        self.model.to(dtype)
        with torch.no_grad():
            res = self.model.infer_single_view(
                image.unsqueeze(0).to(device, dtype),
                None, None,
                render_w2cs=render_w2c.to(device),
                render_intrs=render_intr.to(device),
                render_bg_colors=bg_colors.to(device),
                flame_params={k: v.to(device) for k, v in flame_params.items()}
            )

        # optional video (only meaningful if you have motion)
        if export_video:
            self.model.save_video(
                dump_video_path,
                image.unsqueeze(0).to(device, dtype),
                flame_params={k: v.to(device) for k, v in flame_params.items()},
                intrinsics=render_intr.to(device),
                render_bg_color=bg_colors.to(device)
            )

        # ---------- save predicted RGBs ----------
        rgb = (res['comp_rgb'].clamp(0,1) * 255).cpu().numpy().astype(np.uint8)  # [N,H,W,3]
        if dump_image_dir:
            src_cam = Path(image_path).stem
            src_dir = os.path.join(dump_image_dir, src_cam)
            os.makedirs(src_dir, exist_ok=True)

            for idx in range(N):
                fname = f"{src_cam}_to_{driving_cam_ids[idx]}_{idx:04d}.png"
                Image.fromarray(rgb[idx]).save(os.path.join(src_dir, fname))
                # optional PLY save
                res['3dgs'][idx][0][0].save_ply(os.path.join(src_dir, f"{idx:04d}.ply"))
            dump_cano_dir = "./exps/cano_gs/"
            if not os.path.exists(dump_cano_dir):
                os.system(f"mkdir -p {dump_cano_dir}")
            import trimesh
            vtxs = res['cano_gs_lst'][0].xyz - res['cano_gs_lst'][0].offset
            vtxs = vtxs.detach().cpu().numpy() 
            faces = self.model.renderer.flame_model.faces.detach().cpu().numpy()
            mesh = trimesh.Trimesh(vertices=vtxs, faces=faces)
            mesh.export(os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_shaped_mesh.obj'))

            # Export textured deformed mesh
            import lam.models.rendering.utils.mesh_utils as mesh_utils
            vtxs = res['cano_gs_lst'][0].xyz.detach().cpu()
            faces = self.model.renderer.flame_model.faces.detach().cpu()
            colors = res['cano_gs_lst'][0].shs.squeeze(1).detach().cpu()
            pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + '_textured_mesh.obj')
            print("Save textured mesh to:", pth)
            mesh_utils.save_obj(pth, vtxs, faces, textures=colors, texture_type="vertex")
            # # 3-panel composite for first driving view
            # first_pred = os.path.join(src_dir, f"{src_cam}_to_{driving_cam_ids[0]}_0000.png")
            # if os.path.isfile(first_pred):
            #     pred_im = Image.open(first_pred)
            #     W, H    = pred_im.size
            #     src_im  = Image.open(image_path).resize((W,H))
            #     drv_im  = Image.open(driving_image_path[0]).resize((W,H))
            #     comp = Image.new("RGB", (W*3, H))
            #     comp.paste(src_im, (0,0)); comp.paste(drv_im,(W,0)); comp.paste(pred_im,(2*W,0))
            #     comp.save(first_pred)

        # ---------- metrics ----------
        # gt0 = Image.open(driving_image_path[0]).resize((rgb[0].shape[1], rgb[0].shape[0]))
        # gt_arr = np.array(gt0).astype(np.float32)/255.0
        # gt_t = torch.from_numpy(gt_arr).permute(2,0,1).unsqueeze(0).to(device)

        # metrics = []
        # for idx in range(N):
        #     p = torch.from_numpy(rgb[idx].astype(np.float32)/255.0).permute(2,0,1).unsqueeze(0).to(device)
        #     l1   = self.l1_loss_fn(p, gt_t).item()
        #     lp   = self.lpips_loss_fn(p, gt_t).item()
        #     ps   = self.psnr_metric(p, gt_t).item()
        #     ss   = self.ssim_metric(p, gt_t).item()
        #     metrics.append((idx, l1, lp, ps, ss))

        # return metrics

    def infer(self):
        import numpy as np
        if self.cfg.get('use_cafca_dataset', False):
            target_sid = int(self.cfg.cafca_subject_id_for_single_infer)
            # all frames for that subject
            data     = [d for d in self.cafca_dataset.data if d['subject_id_int'] == target_sid]
            # only those marked as valid sources
            sources  = [d for d in data if d.get('is_source_candidate')]
            drivings = data
        
        for src in tqdm(sources, desc=f"Subject {target_sid} sources"):
            src_cam = src['cam_id']
            is_first = True
            # prepare per-source output dir & CSV path
            img_subdir = os.path.join(self.cfg.image_dump, str(target_sid), src_cam)
            os.makedirs(img_subdir, exist_ok=True)
            csv_path   = os.path.join(img_subdir, "metrics.csv")

            # ── stack all driving intrinsics & world2cam ──────────────────
            drv_intr  = np.stack([d['intrinsic']     for d in drivings])      # [N,4,4]
            drv_w2c   = np.stack([d['world_2_cam']   for d in drivings])      # [N,4,4]
            drv_paths = [d['image_file_path'] for d in drivings]              # list of N paths
            drv_cams  = [d['cam_id']         for d in drivings]

            # only save a video once per source
            vid_dir = os.path.join(self.cfg.video_dump, str(target_sid))
            os.makedirs(vid_dir, exist_ok=True)
            vid_path = os.path.join(vid_dir, f"{src_cam}_to_ALL.mp4")

            # run a single inference call
            self.infer_single(
                image_path                 = src['image_file_path'],
                driving_image_path         = drv_paths[0],
                mask_path_for_preprocess   = src['mask_file_path'],
                target_intrinsics          = drv_intr,
                canonical_flame_path_for_subject = src['canonical_flame_param_path'],
                world2cam                  = drv_w2c,
                dump_image_dir             = img_subdir,
                dump_video_path            = vid_path,
                export_video               = True,
                driving_cam_ids=        drv_cams
            )
