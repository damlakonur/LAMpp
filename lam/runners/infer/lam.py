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
import traceback
import time
import torch
import os
import argparse
import mcubes
import trimesh
import numpy as np
from PIL import Image
from glob import glob
import cv2
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from accelerate.logging import get_logger

from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image, load_flame_params, render_flame_mesh


from .base_inferrer import Inferrer
from lam.datasets.cam_utils import build_camera_principle, build_camera_standard, surrounding_views_linspace, create_intrinsics
from lam.utils.logging import configure_logger
from lam.runners import REGISTRY_RUNNERS
from lam.utils.video import images_to_video
from lam.utils.hf_hub import wrap_model_hub
from lam.models.modeling_lam import ModelLAM
from safetensors.torch import load_file
from lam.dataset import env_paths
from lam.dataset.cafca_lam_dataset import CafcaLamDataset


logger = get_logger(__name__)


def parse_configs():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--infer', type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get('APP_INFER') is not None:
        args.infer = os.environ.get('APP_INFER')
    if os.environ.get('APP_MODEL_NAME') is not None:
        cli_cfg.model_name = os.environ.get('APP_MODEL_NAME')

    if args.config is not None:
        cfg = OmegaConf.load(args.config)
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(cfg_train.experiment.parent, cfg_train.experiment.child, os.path.basename(cli_cfg.model_name).split('_')[-1])

        cfg.save_tmp_dump = os.path.join("exps", 'save_tmp', _relative_path)
        cfg.image_dump = os.path.join("exps", 'images', _relative_path)
        cfg.video_dump = os.path.join("exps", 'videos', _relative_path)
        cfg.mesh_dump = os.path.join("exps", 'meshes', _relative_path)
        
    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault("save_tmp_dump", os.path.join("exps", cli_cfg.model_name, 'save_tmp'))
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, 'images'))
        cfg.setdefault('video_dump', os.path.join("dumps", cli_cfg.model_name, 'videos'))
        cfg.setdefault('mesh_dump', os.path.join("dumps", cli_cfg.model_name, 'meshes'))
        
    
    cfg.motion_video_read_fps = 6
    cfg.merge_with(cli_cfg)
    cfg.setdefault('use_cafca_dataset', False)
    cfg.setdefault('cafca_subject_id_for_single_infer', None) # e.g., 30
    cfg.setdefault('cafca_camera_id_for_single_infer', None)  # e.g., "C02"
    cfg.setdefault('cafca_driving_camera_id_for_single', None)  # e.g., "C02"

    """
    [required]
    model_name: str
    image_input: str
    export_video: bool
    export_mesh: bool

    [special]
    source_size: int
    render_size: int
    video_dump: str
    mesh_dump: str

    [default]
    render_views: int
    render_fps: int
    mesh_size: int
    mesh_thres: float
    frame_size: int
    logger: str
    """

    cfg.setdefault('logger', 'INFO')

    # assert not (args.config is not None and args.infer is not None), "Only one of config and infer should be provided"
    assert cfg.model_name is not None, "model_name is required"
    if not os.environ.get('APP_ENABLED', None) and not cfg.get('use_cafca_dataset', False):
        assert cfg.image_input is not None, "image_input is required"
        assert cfg.export_video or cfg.export_mesh, \
            "At least one of export_video or export_mesh should be True"
        cfg.app_enabled = False
    else:
        cfg.app_enabled = True

    return cfg


@REGISTRY_RUNNERS.register('infer.lam')
class LAMInferrer(Inferrer):

    EXP_TYPE: str = 'lam'

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
            self.cafca_dataset = CafcaLamDataset(subject_list=env_paths.subjects_train, mode="lam_infer")
            self.cafca_loader = torch.utils.data.DataLoader(
                self.cafca_dataset, batch_size=1, shuffle=False, num_workers=0 # Batch size 1 for inference
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

    def _default_source_camera(self, dist_to_center: float = 2.0, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, D_cam_raw)
        canonical_camera_extrinsics = torch.tensor([[
            [1, 0, 0, 0],
            [0, 0, -1, -dist_to_center],
            [0, 1, 0, 0],
        ]], dtype=torch.float32, device=device)
        canonical_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0)
        source_camera = build_camera_principle(canonical_camera_extrinsics, canonical_camera_intrinsics)
        return source_camera.repeat(batch_size, 1)

    def _default_render_cameras(self, n_views: int, batch_size: int = 1, device: torch.device = torch.device('cpu')):
        # return: (N, M, D_cam_render)
        render_camera_extrinsics = surrounding_views_linspace(n_views=n_views, device=device)
        render_camera_intrinsics = create_intrinsics(
            f=0.75,
            c=0.5,
            device=device,
        ).unsqueeze(0).repeat(render_camera_extrinsics.shape[0], 1, 1)
        render_cameras = build_camera_standard(render_camera_extrinsics, render_camera_intrinsics)
        return render_cameras.unsqueeze(0).repeat(batch_size, 1, 1)

    def infer_planes(self, image: torch.Tensor, source_cam_dist: float):
        N = image.shape[0]
        source_camera = self._default_source_camera(dist_to_center=source_cam_dist, batch_size=N, device=self.device)
        planes = self.model.forward_planes(image, source_camera)
        assert N == planes.shape[0]
        return planes


    def infer_mesh(self, planes: torch.Tensor, mesh_size: int, mesh_thres: float, dump_mesh_path: str):
        grid_out = self.model.synthesizer.forward_grid(
            planes=planes,
            grid_size=mesh_size,
        )
        
        vtx, faces = mcubes.marching_cubes(grid_out['sigma'].squeeze(0).squeeze(-1).cpu().numpy(), mesh_thres)
        vtx = vtx / (mesh_size - 1) * 2 - 1

        vtx_tensor = torch.tensor(vtx, dtype=torch.float32, device=self.device).unsqueeze(0)
        vtx_colors = self.model.synthesizer.forward_points(planes, vtx_tensor)['rgb'].squeeze(0).cpu().numpy()  # (0, 1)
        vtx_colors = (vtx_colors * 255).astype(np.uint8)
        
        mesh = trimesh.Trimesh(vertices=vtx, faces=faces, vertex_colors=vtx_colors)

        # dump
        os.makedirs(os.path.dirname(dump_mesh_path), exist_ok=True)
        mesh.export(dump_mesh_path)

    def add_audio_to_video(self, video_path, out_path, audio_path):
        from moviepy.editor import VideoFileClip, AudioFileClip
        video_clip = VideoFileClip(video_path)
        audio_clip = AudioFileClip(audio_path)
        video_clip_with_audio = video_clip.set_audio(audio_clip)
        video_clip_with_audio.write_videofile(out_path, codec='libx264', audio_codec='aac')
        print(f"Audio added successfully at {out_path}")

    def save_imgs_2_video(self, img_lst, v_pth, fps):
        from moviepy.editor import ImageSequenceClip
        images = [image.astype(np.uint8) for image in img_lst]
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(v_pth, codec='libx264')
        print(f"Video saved successfully at {v_pth}")
    
    def infer_single(self, image_path: str,
                     mask_path_for_preprocess: str, # Path to the original mask
                     target_intrinsics: np.ndarray, # Intrinsics for the target image
                     canonical_flame_path_for_subject: str, # Path to subject's canonical_flame_param.npz
                     cam2world: np.ndarray, # Camera to canonical flame transformation
                     motion_seqs_dir, 
                     motion_img_dir,
                     motion_video_read_fps,
                     export_video: bool, 
                     export_mesh: bool, 
                     dump_tmp_dir:str,  # require by extracting motion seq from video, to save some results
                     dump_image_dir:str,
                     dump_video_path: str, 
                     dump_mesh_path: str):
        source_size = self.cfg.source_size
        render_size = self.cfg.render_size
        render_fps = self.cfg.render_fps
        aspect_standard = 1.0/1.0
        motion_img_need_mask = self.cfg.get("motion_img_need_mask", False)  # False
        vis_motion = self.cfg.get("vis_motion", False)  # False
        save_ply = self.cfg.get("save_ply", False)  # False
        save_img = self.cfg.get("save_img", False)  # False
        rendered_bg = 1.
        ref_bg = 1.
        if not os.path.exists(mask_path_for_preprocess):
            logger.warning(f"Mask for preprocess_image not found: {mask_path_for_preprocess}. Relying on rembg if need_mask=True.")
            effective_mask_path = None
        else:
            effective_mask_path = mask_path_for_preprocess

        image, _, _, shape_param = preprocess_image(image_path, mask_path=effective_mask_path, 
                                             intr=None, pad_ratio=0, bg_color=ref_bg, 
                                             max_tgt_size=None, aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
                                             render_tgt_size=source_size, multiply=14, need_mask=True, get_shape_param=True, canonical_flame_path_override=canonical_flame_path_for_subject)
        render_c2ws_single = torch.from_numpy(cam2world).float().unsqueeze(0).unsqueeze(1)
        render_intrs_single = torch.from_numpy(target_intrinsics).float().unsqueeze(0).unsqueeze(1)
        rendered_bg_colors = torch.ones((1, 1, 3), dtype=torch.float32)
        flame_params = load_flame_params(canonical_flame_path_for_subject)
        flame_params["betas"] = shape_param.unsqueeze(0)
        # Reshape pose/expression parameters to be [B, N_source, N_coeffs] = [1, 1, N_coeffs]
        # to match the expectations of infer_single_view.
        params_to_reshape = ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation']
        for p_key in params_to_reshape:
            if p_key in flame_params: # Check if key exists (it should from load_flame_params)
                flame_params[p_key] = flame_params[p_key].unsqueeze(0).unsqueeze(0)
        # motion_render = render_flame_mesh(flame_params, render_intrs_single, render_c2ws_single)
            
        
        start_time = time.time()
        device="cuda"
        dtype=torch.float32
        # dtype=torch.bfloat16
        self.model.to(dtype)
        print("start to inference...................")
        with torch.no_grad():
            # TODO check device and dtype
            res = self.model.infer_single_view(image.unsqueeze(0).to(device, dtype), None, None, 
                                               render_c2ws=render_c2ws_single.to(device),
                                               render_intrs=render_intrs_single.to(device),
                                               render_bg_colors=rendered_bg_colors.to(device),
                                               flame_params={k:v.to(device) for k, v in flame_params.items()})

        breakpoint()
        print(f"time elapsed: {time.time() - start_time}")
        rgb = res["comp_rgb"].detach().cpu().numpy()  # [Nv, H, W, 3], 0-1
        rgb = (np.clip(rgb, 0, 1.0) * 255).astype(np.uint8)
        only_pred = rgb.copy()
        if vis_motion:
            # print(rgb.shape, motion_seq["vis_motion_render"].shape)
            vis_ref_img = np.tile(cv2.resize(vis_ref_img, (rgb[0].shape[1], rgb[0].shape[0]), interpolation=cv2.INTER_AREA)[None, :, :, :], (rgb.shape[0], 1, 1, 1))
            blend_ratio = 0.7
            # blend_res = ((1 -  blend_ratio) * rgb + blend_ratio * motion_seq["vis_motion_render"]).astype(np.uint8)
            # rgb = np.concatenate([rgb, motion_seq["vis_motion_render"], blend_res, vis_ref_img], axis=2)
            rgb = np.concatenate([vis_ref_img, rgb, motion_render], axis=2)
            
        os.makedirs(os.path.dirname(dump_video_path), exist_ok=True)
        # images_to_video(rgb, output_path=dump_video_path, fps=render_fps, gradio_codec=False, verbose=True)
        self.save_imgs_2_video(rgb, dump_video_path, render_fps)
        base_vid = motion_seqs_dir.strip('/').split('/')[-1]
        audio_path = os.path.join(motion_seqs_dir, base_vid+".wav")
        dump_video_path_wa = dump_video_path.replace(".mp4", "_audio.mp4")
        self.add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)
        if save_img and dump_image_dir is not None:
            for i in range(rgb.shape[0]):
                save_file = os.path.join(dump_image_dir, f"{i:04d}.png")
                Image.fromarray(only_pred[i]).save(save_file)
                if save_ply and dump_mesh_path is not None:
                    res["3dgs"][i][0][0].save_ply(os.path.join(dump_image_dir, f"{i:04d}.ply"))

            dump_cano_dir = "./exps/cano_gs/"
            if not os.path.exists(dump_cano_dir):
                os.system(f"mkdir -p {dump_cano_dir}")
            cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + ".ply")
            # res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=True, offset2xyz=False)
            cano_ply_pth = os.path.join(dump_cano_dir, os.path.basename(dump_image_dir) + "_gs_offset.ply")
            res['cano_gs_lst'][0].save_ply(cano_ply_pth, rgb2sh=False, offset2xyz=True)
            # res['cano_gs_lst'][0].save_ply("tmp.ply", rgb2sh=False, offset2xyz=True)

            def save_color_points(points, colors, sv_pth, sv_fd="debug_vis/dataloader/"):
                points = points.squeeze().detach().cpu().numpy()
                colors = colors.squeeze().detach().cpu().numpy()
                sv_pth = os.path.join(sv_fd, sv_pth)
                if not os.path.exists(sv_fd):
                    os.system(f"mkdir -p {sv_fd}")
                with open(sv_pth, 'w') as of:
                    for point, color in zip(points, colors):
                        print('v', point[0], point[1], point[2], color[0], color[1], color[2], file=of)
 
            # save canonical color point clouds
            save_color_points(res['cano_gs_lst'][0].xyz, res["cano_gs_lst"][0].shs[:, 0, :], "framework_img.obj", sv_fd=dump_cano_dir) 

            # Export the template mesh to an OBJ file
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

    def infer(self):
        if self.cafca_loader is not None and self.cfg.get('use_cafca_dataset', False):
            logger.info("Inferring using CafcaLamDataset.")
            input_source_is_cafca = True
        elif os.path.isfile(self.cfg.image_input):
            logger.info(f"Inferring using single image_input: {self.cfg.image_input}")
            input_source_is_cafca = False
        if input_source_is_cafca and self.cfg.get('cafca_subject_id_for_single_infer') is not None and self.cfg.get('cafca_camera_id_for_single_infer') is not None and self.cfg.get('cafca_driving_camera_id_for_single') is not None:
            target_subject_id = int(self.cfg.cafca_subject_id_for_single_infer)
            target_cam_id = self.cfg.cafca_camera_id_for_single_infer
            logger.info(f"Attempting to find specific CAFCA item: Subject ID {target_subject_id}, Camera ID {target_cam_id}")
            target_driving_cam_id = self.cfg.cafca_driving_camera_id_for_single
            
            found_item = None
            found_driving_item = None
            for item in self.cafca_dataset.data: # Accessing internal list directly for searching
                if item["subject_id_int"] == target_subject_id and item["cam_id"] == target_cam_id:
                    found_item = item
                if item["subject_id_int"] == target_subject_id and item["cam_id"] == target_driving_cam_id:
                    found_driving_item = item
            if found_item is None:
                logger.error(f"Could not find CAFCA item for Subject ID {target_subject_id}, Camera ID {target_cam_id}. Exiting.")
                return
            iterable_data_for_loop = [{k: [v] for k,v in found_item.items()}]

        target_cam2world = found_driving_item["cam_2_world"]
        target_intrinsics = found_driving_item["intrinsic"]
        if isinstance(target_cam2world, torch.Tensor):
            target_cam2world = target_cam2world.cpu().numpy()
            
        

        for data_item_batch in tqdm(iterable_data_for_loop, disable=not self.accelerator.is_local_main_process):
            try:
                if input_source_is_cafca:
                    # Item from CafcaLamDataset DataLoader (batch size 1)
                    ref_preprocessed_image_path = data_item_batch["image_file_path"][0]
                    ref_preprocessed_mask_path = data_item_batch["mask_file_path"][0]
                    # Intrinsics are already numpy arrays from CafcaLamDataset
                    ref_canonical_flame_path = data_item_batch["canonical_flame_param_path"][0]
                    subject_id_str_ref = data_item_batch["subject_id_str"][0]
                    cam_id_ref = data_item_batch["cam_id"][0]
                    ref_cam2world = data_item_batch["cam_2_world"][0]
                    if isinstance(ref_cam2world, torch.Tensor):
                        ref_cam2world = ref_cam2world.cpu().numpy()
                    
                    
                    uid = f"{subject_id_str_ref}_{cam_id_ref}"
                    dump_subdir = subject_id_str_ref

                motion_seqs_dir = self.cfg.motion_seqs_dir
                print("motion_seqs_dir:", motion_seqs_dir)

                dump_video_path = os.path.join(
                    self.cfg.video_dump,
                    dump_subdir,
                    f'{uid}.mp4',
                )
                dump_image_dir = os.path.join(
                    self.cfg.image_dump,
                    dump_subdir,
                    f'{uid}'
                )
                dump_tmp_dir = os.path.join(
                    self.cfg.image_dump,
                    dump_subdir,
                    "tmp_res"
                )
                dump_mesh_path = os.path.join(
                    self.cfg.mesh_dump,
                    dump_subdir,
                    # f'{uid}.ply',
                )
                os.makedirs(dump_image_dir, exist_ok=True)
                os.makedirs(dump_tmp_dir, exist_ok=True)
                os.makedirs(dump_mesh_path, exist_ok=True)

                # if os.path.exists(dump_video_path):
                #     print(f"skip:{image_path}")
                #     continue

                self.infer_single(
                    image_path=ref_preprocessed_image_path, # This is the 512x512 image
                    mask_path_for_preprocess=ref_preprocessed_mask_path,
                    target_intrinsics=target_intrinsics,
                    canonical_flame_path_for_subject=ref_canonical_flame_path,
                    cam2world=target_cam2world,
                    motion_seqs_dir=motion_seqs_dir,
                    motion_img_dir=self.cfg.motion_img_dir,
                    motion_video_read_fps=self.cfg.motion_video_read_fps,
                    export_video=self.cfg.export_video,
                    export_mesh=self.cfg.export_mesh, 
                    dump_tmp_dir=dump_tmp_dir,
                    dump_image_dir=dump_image_dir,
                    dump_video_path=dump_video_path,
                    dump_mesh_path=dump_mesh_path,
                    )
            except:
                traceback.print_exc()
