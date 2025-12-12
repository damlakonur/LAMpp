# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors. 
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
# limitations under the License.

import os
import torch.nn.functional as F
import math
import pyvista as pv
import trimesh
import cv2
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRasterizer
)
from collections import defaultdict
import torch.nn as nn
import numpy as np
import torch
import torch.nn as nn
from accelerate.logging import get_logger
from einops import rearrange, repeat

from .transformer import TransformerDecoder
from lam.models.rendering.gs_renderer import GS3DRenderer, PointEmbed
from diffusers.utils import is_torch_version
from mediapy import VideoWriter
from dreifus.trajectory import circle_around_axis
from dreifus.vector import Vec3
from tqdm import tqdm
import os
from dreifus.matrix import Intrinsics, Pose
from lam.models.rendering.utils.mesh_utils import axis_angle_to_matrix

logger = get_logger(__name__)

def extract_camera_centers(source_view_w2cs):
    # Extract rotation and translation explicitly
    R = source_view_w2cs[..., :3, :3]       # [...,3,3]
    t = source_view_w2cs[..., :3, 3]        # [...,3]   ←  **note :3, 3  (not 3:)**

    C = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1)   # [...,3]
    return C


class ModelLAM(nn.Module):
    """
    Full model of the basic single-view large reconstruction model.
    """
    def __init__(self,
                 transformer_dim: int, transformer_layers: int,  transformer_heads: int,
                 transformer_type="cond",
                 tf_grad_ckpt=False,
                 encoder_grad_ckpt=False,
                 encoder_freeze: bool = True, encoder_type: str = 'dino',
                 encoder_model_name: str = 'facebook/dino-vitb16', encoder_feat_dim: int = 768,
                 num_pcl: int=2048, pcl_dim: int=512,
                 human_model_path="./model_zoo/human_parametric_models",
                 flame_subdivide_num=2,
                 flame_type="flame",
                 gs_query_dim=None,
                 gs_use_rgb=False,
                 gs_sh=3,
                 gs_mlp_network_config=None,
                 gs_xyz_offset_max_step=1.8 / 32,
                 gs_clip_scaling=0.2,
                 shape_param_dim=100,
                 expr_param_dim=50,
                 fix_opacity=False,
                 fix_rotation=False,
                 num_source_views=1,
                 flame_scale=1.0,
                 instantiate_encoder=False,
                 instantiate_transformer=False,
                 num_gaussians_per_vertex=1,
                 is_finetuning=False,
                 **kwargs,
                 ):
        super().__init__()
        self.gradient_checkpointing = tf_grad_ckpt
        self.encoder_gradient_checkpointing = encoder_grad_ckpt
        
        # attributes
        self.encoder_feat_dim = encoder_feat_dim
        self.conf_use_pred_img = False
        self.conf_cat_feat = False and self.conf_use_pred_img  # True # False
        self.num_source_views = num_source_views
        self.flame_scale = flame_scale

        # modules
        # image encoder
        if instantiate_encoder:
            self.encoder = self._encoder_fn(encoder_type)(
                model_name=encoder_model_name,
                freeze=encoder_freeze,
                encoder_feat_dim=encoder_feat_dim,
            )
        else:
            self.encoder = None

        # learnable points embedding
        skip_decoder = False
        self.latent_query_points_type = kwargs.get("latent_query_points_type", "e2e_flame")
        if self.latent_query_points_type == "embedding":
            self.num_pcl = num_pcl
            self.pcl_embeddings = nn.Embedding(num_pcl , pcl_dim)
        elif self.latent_query_points_type.startswith("flame"):
            latent_query_points_file = os.path.join(human_model_path, "flame_points", f"{self.latent_query_points_type}.npy")
            pcl_embeddings = torch.from_numpy(np.load(latent_query_points_file)).float()
            print(f"==========load flame points:{latent_query_points_file}, shape:{pcl_embeddings.shape}")
            self.register_buffer("pcl_embeddings", pcl_embeddings)
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        elif self.latent_query_points_type.startswith("e2e_flame"):
            skip_decoder = True
            self.pcl_embed = PointEmbed(dim=pcl_dim)
        else:
            raise NotImplementedError
        print("==="*16*3, f"\nskip_decoder: {skip_decoder}", "\n"+"==="*16*3)
        # transformer
        if instantiate_transformer:
            self.transformer = TransformerDecoder(
                block_type=transformer_type,
                num_layers=transformer_layers, num_heads=transformer_heads,
                inner_dim=transformer_dim, cond_dim=encoder_feat_dim, mod_dim=None,
                gradient_checkpointing=self.gradient_checkpointing,
            )
        else:
            self.transformer = None
            
        # To fuse information from n-views
        print(f"[DEBUG] num_source_views: {self.num_source_views}")
        if self.num_source_views > 1:
            # initialzie it with zeros
            # self.layer_norm = nn.LayerNorm(transformer_dim)
            # self.fusion_layer = nn.Linear(transformer_dim * self.num_source_views, transformer_dim)
            # nn.init.zeros_(self.fusion_layer.weight)
            # nn.init.zeros_(self.fusion_layer.bias)
            
            # Method 2: Weighting-based fusion
            # hidden_dim = transformer_dim // 4  
            # self.view_weighting = nn.Sequential(
            #     nn.Linear(transformer_dim, hidden_dim),
            #     nn.SiLU(),
            #     nn.Linear(hidden_dim, 1)  # scalar weight per point per view
            # )
            # self.fusion_layer = nn.Linear(transformer_dim, transformer_dim)
            # nn.init.zeros_(self.fusion_layer.weight)
            # nn.init.zeros_(self.fusion_layer.bias)
            #########################################################################################

            # ------------------------------------------------------------------ #
            # New: MLP-based fusion that leverages visibility scores and Plücker
            #      coordinates, followed by a self-attention refinement.
            # ------------------------------------------------------------------ #
            concat_dim = self.num_source_views * (transformer_dim + 6 + 1)  # latent + plücker(6) + vis(1)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(concat_dim, transformer_dim),
                nn.SiLU(),
                nn.Linear(transformer_dim, transformer_dim),
            )
            
            # # Cross-attention layer after fusion
            # print(f"[DEBUG] Creating post_fusion_transformer with:")
            # print(f"  - transformer_dim: {transformer_dim}")
            # print(f"  - transformer_heads: {transformer_heads}")
            # print(f"  - encoder_feat_dim: {encoder_feat_dim}")
            # print(f"  - cond_dim: {encoder_feat_dim}")
            
            # self.post_fusion_transformer = TransformerDecoder(
            #     block_type="sd3_cond",
            #     num_layers=1,  # Only one layer for cross-attention
            #     num_heads=transformer_heads,
            #     inner_dim=transformer_dim, 
            #     cond_dim=encoder_feat_dim,  # Each image feature has encoder_feat_dim, don't multiply by 2
            #     mod_dim=None,
            #     gradient_checkpointing=self.gradient_checkpointing,
            # )
            # print(f"[DEBUG] post_fusion_transformer created successfully")
            
            # # Initialize the transformer to output zeros for stable training start
            # self._initialize_transformer_to_zero()
            
            # # Count parameters in the new cross-attention layer
            # ca_params = sum(p.numel() for p in self.post_fusion_transformer.parameters() if p.requires_grad)
            # print(f"[DEBUG] Cross-attention layer has {ca_params:,} trainable parameters")
            
            # Self-attention layer after fusion (simple nn.MultiHeadAttention)
            # print(f"[DEBUG] Creating post_fusion_self_attention with:")
            # print(f"  - transformer_dim: {transformer_dim}")
            # print(f"  - transformer_heads: {transformer_heads}")
            
            # # Use MultiheadAttention without batch_first for compatibility
            # self.post_fusion_self_attention = nn.MultiheadAttention(
            #     embed_dim=transformer_dim,
            #     num_heads=32,
            #     dropout=0.0,
            #     bias=True
            # )
            
            # Add layer normalization for stability
            # self.pre_attention_norm = nn.LayerNorm(transformer_dim)
            # self.post_attention_norm = nn.LayerNorm(transformer_dim)
            # print(f"[DEBUG] post_fusion_self_attention created successfully")
            
            # # Initialize the self-attention to output zeros for stable training start
            # self._initialize_multihead_attention_to_zero()
            
            # # Count parameters in the new self-attention layer
            # sa_params = sum(p.numel() for p in self.post_fusion_self_attention.parameters() if p.requires_grad)
            
            # print(f"[DEBUG] Self-attention layer has {sa_params:,} trainable parameters")
            # self._initialize_fusion_mlp_to_zero()
            
        
        # renderer
        self.renderer = GS3DRenderer(human_model_path=human_model_path,
                                     subdivide_num=flame_subdivide_num,
                                     smpl_type=flame_type,
                                     feat_dim=transformer_dim,
                                     query_dim=gs_query_dim,
                                     use_rgb=gs_use_rgb,
                                     sh_degree=gs_sh,
                                     mlp_network_config=gs_mlp_network_config,
                                     xyz_offset_max_step=gs_xyz_offset_max_step,
                                     clip_scaling=gs_clip_scaling,
                                     scale_sphere=kwargs.get("scale_sphere", False),
                                     shape_param_dim=shape_param_dim,
                                     expr_param_dim=expr_param_dim,
                                     fix_opacity=fix_opacity,
                                     fix_rotation=fix_rotation,
                                     skip_decoder=skip_decoder,
                                     decode_with_extra_info=kwargs.get("decode_with_extra_info", None),
                                     gradient_checkpointing=self.gradient_checkpointing,
                                     add_teeth=kwargs.get("add_teeth", True),
                                     teeth_bs_flag=kwargs.get("teeth_bs_flag", False),
                                     oral_mesh_flag=kwargs.get("oral_mesh_flag", False),
                                     use_mesh_shading=kwargs.get('use_mesh_shading', False),
                                     render_rgb=kwargs.get("render_rgb", True),
                                     num_gaussians_per_vertex=num_gaussians_per_vertex,
                                     is_finetuning=is_finetuning,
                                     )

    def get_last_layer(self):
        return self.renderer.gs_net.out_layers["shs"].weight
    
    def _initialize_transformer_to_zero(self):
        """Initialize the post-fusion transformer to output zeros for stable training start."""
        if hasattr(self, 'post_fusion_transformer') and self.post_fusion_transformer is not None:
            # Zero out the final output projection to make the transformer output zeros initially
            for layer in self.post_fusion_transformer.layers:
                # Handle SD3 transformer structure - zero out the final output projection
                if hasattr(layer, 'attn') and hasattr(layer.attn, 'to_out'):
                    to_out = layer.attn.to_out
                    # Handle different types of output layers
                    if isinstance(to_out, nn.Sequential):
                        # Sequential case: zero the first Linear layer
                        for module in to_out:
                            if isinstance(module, nn.Linear):
                                nn.init.zeros_(module.weight)
                                if module.bias is not None:
                                    nn.init.zeros_(module.bias)
                                break
                    elif isinstance(to_out, nn.ModuleList):
                        # ModuleList case: zero the first Linear layer in the list
                        for module in to_out:
                            if isinstance(module, nn.Linear):
                                nn.init.zeros_(module.weight)
                                if module.bias is not None:
                                    nn.init.zeros_(module.bias)
                                break
                    elif isinstance(to_out, nn.Linear):
                        # Direct Linear layer
                        nn.init.zeros_(to_out.weight)
                        if to_out.bias is not None:
                            nn.init.zeros_(to_out.bias)
                
                # Also zero out the feedforward output if it exists
                if hasattr(layer, 'ff') and hasattr(layer.ff, 'net'):
                    # Find the last linear layer in the feedforward network
                    for module in reversed(list(layer.ff.net.modules())):
                        if isinstance(module, nn.Linear):
                            nn.init.zeros_(module.weight)
                            if module.bias is not None:
                                nn.init.zeros_(module.bias)
                            break
            
            print("[DEBUG] Initialized post_fusion_transformer to output zeros")
    
    def _initialize_multihead_attention_to_zero(self):
        """Initialize the MultiHeadAttention to output zeros for stable training start."""
        if hasattr(self, 'post_fusion_self_attention') and self.post_fusion_self_attention is not None:
            # Zero out the output projection to make the attention output zeros initially
            if hasattr(self.post_fusion_self_attention, 'out_proj'):
                nn.init.zeros_(self.post_fusion_self_attention.out_proj.weight)
                if self.post_fusion_self_attention.out_proj.bias is not None:
                    nn.init.zeros_(self.post_fusion_self_attention.out_proj.bias)
            
            print("[DEBUG] Initialized MultiHeadAttention to output zeros")
            
    def _initialize_fusion_mlp_to_zero(self):
        """Initialize the fusion MLP to output zeros for stable training start."""
        if hasattr(self, 'fusion_mlp') and self.fusion_mlp is not None:
            for module in self.fusion_mlp:
                if isinstance(module, nn.Linear):
                    nn.init.zeros_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            print("[DEBUG] Initialized fusion MLP to output zeros")
            
    @staticmethod
    def _encoder_fn(encoder_type: str):
        from .encoders.dinov2_fusion_wrapper import Dinov2FusionWrapper
        return Dinov2FusionWrapper
        
    def forward_transformer(self, image_feats, camera_embeddings, query_points, query_feats=None):
        # assert image_feats.shape[0] == camera_embeddings.shape[0], \
        #     "Batch size mismatch for image_feats and camera_embeddings!"
        B = image_feats.shape[0]
        # Attaches learnable features to the flame vertices to use in cross attention
        if self.latent_query_points_type == "embedding":
            range_ = torch.arange(self.num_pcl, device=image_feats.device)
            x =  self.pcl_embeddings(range_).unsqueeze(0).repeat((B, 1, 1)) # [B, L, D]
            
        elif self.latent_query_points_type.startswith("flame"):
            x = self.pcl_embed(self.pcl_embeddings.unsqueeze(0)).repeat((B, 1, 1)) # [B, L, D]

        elif self.latent_query_points_type.startswith("e2e_flame"):
            x = self.pcl_embed(query_points) # [B, L, D]

        x = x.to(image_feats.dtype)
        if query_feats is not None:
            x = x + query_feats.to(image_feats.dtype)
        x = self.transformer(
            x,
            cond=image_feats,
            mod=camera_embeddings,
        )  # [B, L, D]
        # x = x.to(image_feats.dtype)
        return x

    def forward_encode_image(self, image):
        image = image.to(dtype=torch.float32)
        # encode image
        if self.training and self.encoder_gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward
            ckpt_kwargs = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            image_feats = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.encoder),
                image,
                **ckpt_kwargs,
            )
        else:
            image_feats = self.encoder(image)
        return image_feats

    # @torch.compile
    def forward_latent_points(self, image, camera, query_points=None, additional_features=None):
        # image: [B, C_img, H_img, W_img]
        # camera: [B, D_cam_raw]
        B = image.shape[0]

        # encode image
        image_feats = self.forward_encode_image(image)
        
        assert image_feats.shape[-1] == self.encoder_feat_dim, \
            f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"

        if additional_features is not None and len(additional_features.keys()) > 0:
            image_feats_bchw = rearrange(image_feats, "b (h w) c -> b c h w", h=int(math.sqrt(image_feats.shape[1])))
            additional_features["source_image_feats"] = image_feats_bchw
            proj_feats = self.renderer.get_batch_project_feats(None, query_points, additional_features=additional_features, feat_nms=['source_image_feats'], use_mesh=True)
            query_feats = proj_feats['source_image_feats']
        else:
            query_feats = None

        # transformer generating latent points
        # TODO first save directly the tokens then load it, check with profiler or look at it/s
        tokens = self.forward_transformer(image_feats, camera_embeddings=None, query_points=query_points, query_feats=query_feats)

        return tokens, image_feats

    def vis_mask_rasterizer(
            self,
            verts,                  # (B, N, 3)
            faces,                  # (F, 3)
            cam_R, cam_T, K,        # (B, V, 3, 3)  or (B, V, 4, 4)
            image_size=(256, 256),
    ):
        device = verts.device
        N       = verts.shape[1]
        num_faces = faces.shape[0]
        H, W = image_size
        B, V, _ = cam_T.shape

        eye44 = torch.eye(4, device=device, dtype=verts.dtype)
        K44   = eye44.view(1, 1, 4, 4).repeat(B, V, 1, 1)
        K44[:, :, :3, :3] = K.float()
        K44[..., 0,0] *= 0.5
        K44[..., 1,1] *= 0.5
        K44[..., 0,2] *= 0.5
        K44[..., 1,2] *= 0.5


        # ------------------------------------------------------------------ #
        with torch.cuda.amp.autocast(False):
            mesh_batch = Meshes(verts=verts, faces=faces.unsqueeze(0).repeat(B, 1, 1))
            faces_packed   = mesh_batch.faces_packed()      # (B*F, 3)
            first_idx_face = mesh_batch.mesh_to_faces_packed_first_idx()  # (B,)
            first_idx_vert = mesh_batch.mesh_to_verts_packed_first_idx()

            rasteriser = MeshRasterizer(
                raster_settings = RasterizationSettings(
                    image_size      = image_size,
                    faces_per_pixel = 1,
                    blur_radius     = 0.0,
                    max_faces_per_bin=128_000
                )
            )
            vis = torch.zeros((B, V, N), dtype=torch.bool, device=device)

            for v in range(V):
                cameras = PerspectiveCameras(
                    R          = cam_R[:, v].float(),
                    T          = cam_T[:, v].float(),
                    K          = K44[:, v].float(),
                    image_size = torch.tensor([H, W], device=device).repeat(B, 1),
                    in_ndc     = False,
                    device     = device,
                )

                frags = rasteriser(mesh_batch, cameras=cameras)
                pix_to_face = frags.pix_to_face[..., 0]

                # ---- gather per-mesh visible vertices --------------------------------
                for b in range(B):
                    f_start = first_idx_face[b]
                    f_end   = f_start + num_faces

                    face_ids = pix_to_face[b].unique()
                    face_ids = face_ids[(face_ids >= f_start) & (face_ids < f_end)]
                    if face_ids.numel() == 0:
                        continue
                    v_ids = faces_packed[face_ids].view(-1).unique()
                    v_ids_local  = v_ids - first_idx_vert[b]
                    vis[b, v, v_ids_local] = True

        return vis
    

    def save_visibility_snapshot(
            self,
            mesh_pv,        # PyVista PolyData of **the first mesh**
            verts,          # (20018, 3)  torch or numpy – first mesh's verts
            ray_vis0,       # torch.bool (2, 20018) – first mesh, two views
            cam_pos,
            out_png="vis_c00_c06.png",
            window=(512, 512),
    ):
        """
        Colours vertices:
            red     – visible only from C00  (ray_vis0[0])
            blue    – visible only from C06  (ray_vis0[1])
            magenta – visible from both
            grey    – from neither
        and saves an off-screen PNG.
        """
        # w_raw_bool = front_mask * ray_vis.float()
        # verts = query_points[0].cpu().numpy()
        # faces = faces.cpu().numpy()
        # faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
        # mesh = pv.PolyData(verts, faces_pv)
        # self.save_visibility_snapshot(
        #     mesh_pv      = mesh,     # pre-loaded PolyData
        #     verts     = query_points[0],       # (N,3) torch
        #     ray_vis0   = w_raw_bool[0],               # (B,V,N) torch.bool
        #     cam_pos= cam_centers[0, 0].detach().cpu().numpy() ,
        #     out_png   = "visibility_step4.png",
        # )
        # self.save_visibility_snapshot(
        #     mesh_pv      = mesh,     # pre-loaded PolyData
        #     verts     = query_points[1],       # (N,3) torch
        #     ray_vis0   = w_raw_bool[1],               # (B,V,N) torch.bool
        #     cam_pos= cam_centers[1, 0].detach().cpu().numpy() ,
        #     out_png   = "visibility_step5.png",
        # )

        # breakpoint()

        # 1. convert to numpy and ensure boolean type
        vis_c00 = ray_vis0[0].cpu().numpy().astype(bool)
        vis_c06 = ray_vis0[1].cpu().numpy().astype(bool)

        if torch.is_tensor(verts):
            verts = verts.cpu().numpy()

        # 2. colour array
        colours = np.full((verts.shape[0], 3), 0.5, dtype=np.float32)  # grey
        colours[vis_c06]         = [1.0, 0.0, 0.0]   # red
        colours[vis_c00]         = [0.0, 0.0, 1.0]   # blue
        colours[vis_c00 & vis_c06] = [1.0, 0.0, 1.0] # magenta

        # 3. render
        p = pv.Plotter(off_screen=True, window_size=window)
        p.set_background("white")
        p.add_mesh(mesh_pv, color="lightgray", opacity=0.25, show_edges=False)
        # Squeeze the extra dimension from cam_pos and create camera position in PyVista format
        camera_pos = [(float(cam_pos[0,0]), float(cam_pos[0,1]), float(cam_pos[0,2])),  # camera position
                     (0.0, 0.0, 0.0),                                                     # focal point at origin
                     (0.0, 1.0, 0.0)]                                                     # view up vector
        p.camera_position = camera_pos
        p.add_points(
            verts,
            scalars=colours,
            rgb=True,
            point_size=6,
            render_points_as_spheres=True,
        )
        p.show(screenshot=out_png)
        print(f"[✓] wrote {out_png}")

    def forward3(self, src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, latent_points):
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = 512, 512
        query_points = None
        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params, surf_normals, faces = self.renderer.get_query_points(flame_params,
                                                                        device=render_w2cs.device)
        # #### The following part is important to obtain canonical-cam-params #####
        src_w2cs = flame_params["canon_2_cam"][:, :2] @ src_w2cs       
        #########################################################################

        R_world2cam = src_w2cs[:, :, :3, :3]           # (B,V,3,3)
        T_world2cam = src_w2cs[:, :, :3, 3]            # (B,V,3)
        # cam_centers = src_w2cs[:, :, :3, 3] # when using cam2world
        ############ Visibility according to vertex normals #####################
        cam_centers = extract_camera_centers(src_w2cs).unsqueeze(2)    # [B, V, 1, 3]
        vertex_positions = query_points.unsqueeze(1)   # [B, 1, N, 3]
        vertex_normals = surf_normals.unsqueeze(1)     # [B, 1, N, 3]
        # rays from vertex toward camera
        ray_dirs = cam_centers - vertex_positions      # [B, V, N, 3]
        ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=-1)
        # front-facing mask
        cos_theta  = (ray_dirs * vertex_normals).sum(-1)   # [B, V, N]
        front_mask = (cos_theta > 0).float() 
        ############### Visibility according to mesh rasterization ##############
        ray_vis = self.vis_mask_rasterizer(   # [B,V,N]  
                    verts=query_points,
                    faces=faces,
                    cam_R=R_world2cam,
                    cam_T=T_world2cam.float(),
                    K=src_intrs)
        # 2. final per-vertex, per-view weight
        w_raw = front_mask * ray_vis.float()                # [B,V,N]
        # Geometric Visibility Weighting    
        visible_mask   = (w_raw > 0)                       # [B,V,N]
        count_visible  = visible_mask.sum(dim=1, keepdim=True)  # [B,1,N]

        V = w_raw.shape[1]
        w_fuse = torch.where(
            (count_visible == 0) | (count_visible == V),
            torch.full_like(w_raw, 1.0 / V),               # 50% - 50%
            visible_mask.float(),                          # 1 / 0
        )

        latent_points = (w_fuse.unsqueeze(-1) * latent_points).sum(dim=1)  # [B,N,D]

        render_results = self.renderer(gs_hidden_features=latent_points,
                                       query_points=query_points,
                                       flame_data=flame_params,
                                       w2c=render_w2cs,
                                       intrinsic=render_intrs,
                                       height=render_h,
                                       width=render_w,
                                       background_color=render_bg_colors,
                                       additional_features=None
        )

        N, M = render_w2cs.shape[:2]
        assert render_results['comp_rgb'].shape[0] in [N, N], "Batch size mismatch for render_results"
        assert render_results['comp_rgb'].shape[1] in [M, M*2], "Number of rendered views should be consistent with render_cameras"

        return {
            # 'latent_points': latent_points,
            **render_results,
        }

    def forward_img_feats(self, src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, img_feats):
        """
        Forward function that takes img_feats and produces latent points on-the-fly,
        then uses forward1 for fusion and rendering.
        
        Args:
            src_w2cs: Source camera extrinsics [B, V, 4, 4]
            src_intrs: Source camera intrinsics [B, V, 3, 3] 
            render_w2cs: Render camera extrinsics [B, M, 4, 4]
            render_intrs: Render camera intrinsics [B, M, 3, 3]
            render_bg_colors: Background colors [B, M, 3]
            flame_params: FLAME parameters dict
            img_feats: Image features [B, V, H*W, C] from DinoV2
        """
        assert len(flame_params["betas"].shape) == 2
        device = src_w2cs.device
        
        # Step 1: Get FLAME query points (same as forward_latent_points)
        query_points = None
        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params, surf_normals, faces = self.renderer.get_query_points(
                flame_params, device=device)
        
        # Step 2: Generate latent points for each source view (simplified - no additional features)
        # B, V = img_feats.shape[:2]
        # all_latent_points = []
        
        # for v in range(V):
        #     # Get single view image features [B, H*W, C] 
        #     image_feats = img_feats[:, v]  
            
        #     assert image_feats.shape[-1] == self.encoder_feat_dim, \
        #         f"Feature dimension mismatch: {image_feats.shape[-1]} vs {self.encoder_feat_dim}"
            
        #     # Keep it simple - no additional features, just set query_feats = None
        #     query_feats = None
            
        #     # Generate latent points via transformer
        #     tokens = self.forward_transformer(image_feats, camera_embeddings=None, query_points=query_points, query_feats=query_feats)
        #     all_latent_points.append(tokens)
        
        # # Step 3: Stack latent points from all source views [B, V, N, C]
        # latent_points = torch.stack(all_latent_points, dim=1)
        B, V, S, C = img_feats.shape
        image_feats_flat = img_feats.view(B * V, S, C)
        query_points_flat = None
        if query_points is not None:
            query_points_flat = query_points.repeat_interleave(V, dim=0)

        tokens = self.forward_transformer(
            image_feats_flat,
            camera_embeddings=None,
            query_points=query_points_flat,
            query_feats=None,
        )  # [B*V, N, C]

        latent_points = tokens.view(B, V, *tokens.shape[1:])  # [B, V, N, C]
        
        # Step 4: Use forward1 for fusion and rendering
        return self.forward1(src_w2cs, src_intrs, render_w2cs, render_intrs, 
                           render_bg_colors, flame_params, latent_points)
        
    def forward_avg(self, src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, latent_points):
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = 512, 512
        query_points = None
        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params, surf_normals, faces = self.renderer.get_query_points(flame_params,
                                                                        device=render_w2cs.device)
        if latent_points.ndim >= 3 and self.num_source_views > 1:
            avg_points = latent_points.mean(dim=1, keepdim=False)
            latent_points = avg_points
            

        elif self.num_source_views == 1:
            latent_points = latent_points.squeeze(1)

        render_results = self.renderer(gs_hidden_features=latent_points,
                                       query_points=query_points,
                                       flame_data=flame_params,
                                       w2c=render_w2cs,
                                       intrinsic=render_intrs,
                                       height=render_h,
                                       width=render_w,
                                       background_color=render_bg_colors,
                                       additional_features=None
        )

        N, M = render_w2cs.shape[:2]
        assert render_results['comp_rgb'].shape[0] in [N, N], "Batch size mismatch for render_results"
        assert render_results['comp_rgb'].shape[1] in [M, M*2], "Number of rendered views should be consistent with render_cameras"

        return {
            # 'latent_points': latent_points,
            **render_results,
        }
        
    
    def forward1(self, src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, latent_points):
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = 512, 512
        query_points = None
        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params, surf_normals, faces = self.renderer.get_query_points(flame_params,
                                                                        device=render_w2cs.device)
        if latent_points.ndim >= 3 and self.num_source_views > 1:
            avg_points = latent_points.mean(dim=1, keepdim=False)
            
            ############### first method #############
            latent_points = latent_points.float() 
            latent_points = latent_points.permute(0, 2, 1, 3).reshape(latent_points.size(0), latent_points.size(2), -1)
            latent_points = self.fusion_layer(latent_points)
            latent_points += avg_points
            

        elif self.num_source_views == 1:
            latent_points = latent_points.squeeze(1)

        render_results = self.renderer(gs_hidden_features=latent_points,
                                       query_points=query_points,
                                       flame_data=flame_params,
                                       w2c=render_w2cs,
                                       intrinsic=render_intrs,
                                       height=render_h,
                                       width=render_w,
                                       background_color=render_bg_colors,
                                       additional_features=None
        )

        N, M = render_w2cs.shape[:2]
        assert render_results['comp_rgb'].shape[0] in [N, N], "Batch size mismatch for render_results"
        assert render_results['comp_rgb'].shape[1] in [M, M*2], "Number of rendered views should be consistent with render_cameras"

        return {
            # 'latent_points': latent_points,
            **render_results,
        }

    def forward2(self, src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, latent_points):
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = 512, 512
        query_points = None
        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params, surf_normals, faces = self.renderer.get_query_points(flame_params,
                                                                        device=render_w2cs.device)
        if latent_points.ndim >= 3 and self.num_source_views > 1:
            avg_points = latent_points.mean(dim=1, keepdim=False)
            
            ############### second method #############
            latent_points = latent_points.float() 
            weights = self.view_weighting(latent_points) # [B, 2, N, 1]
            weights = F.softmax(weights, dim=1) # softmax across views per-point
            aggregated_points = (latent_points * weights).sum(dim=1)
            latent_points = self.fusion_layer(aggregated_points) + avg_points
            # latent_points = self.layer_norm(latent_points).squeeze(1)
            

        elif self.num_source_views == 1:
            latent_points = latent_points.squeeze(1)

        render_results = self.renderer(gs_hidden_features=latent_points,
                                       query_points=query_points,
                                       flame_data=flame_params,
                                       w2c=render_w2cs,
                                       intrinsic=render_intrs,
                                       height=render_h,
                                       width=render_w,
                                       background_color=render_bg_colors,
                                       additional_features=None
        )

        N, M = render_w2cs.shape[:2]
        assert render_results['comp_rgb'].shape[0] in [N, N], "Batch size mismatch for render_results"
        assert render_results['comp_rgb'].shape[1] in [M, M*2], "Number of rendered views should be consistent with render_cameras"

        return {
            # 'latent_points': latent_points,
            **render_results,
        }
        
    def forward4(self, src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, latent_points, src_canon_2_cam, image_feats=None, source_flame_params=None, render_images=None, data=None):
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = 512, 512
        query_points = None
        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params, surf_normals, faces = self.renderer.get_query_points(flame_params,
                                                                        device=render_w2cs.device)
        if latent_points.ndim >= 3 and self.num_source_views > 1:
            # #### The following part is important to obtain canonical-cam-params #####
            src_w2cs = src_canon_2_cam @ src_w2cs        
            #########################################################################
            R_world2cam = src_w2cs[:, :, :3, :3]           # (B,V,3,3)
            T_world2cam = src_w2cs[:, :, :3, 3]            # (B,V,3)
            # cam_centers = src_w2cs[:, :, :3, 3] # when using cam2world
            ############ Visibility according to vertex normals #####################
            cam_centers = extract_camera_centers(src_w2cs).unsqueeze(2)    # [B, V, 1, 3]
            vertex_positions = query_points.unsqueeze(1)   # [B, 1, N, 3]
            vertex_normals = surf_normals.unsqueeze(1)     # [B, 1, N, 3]
            # rays from vertex toward camera
            ray_dirs = cam_centers - vertex_positions      # [B, V, N, 3]
            ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=-1)
            # front-facing mask
            cos_theta  = (ray_dirs * vertex_normals).sum(-1)   # [B, V, N]
            front_mask = (cos_theta > 0).float() 
            ############### Visibility according to mesh rasterization ##############
            ray_vis = self.vis_mask_rasterizer(   # [B,V,N]  
                        verts=query_points,
                        faces=faces,
                        cam_R=R_world2cam,
                        cam_T=T_world2cam.float(),
                        K=src_intrs)
            # 2. final per-vertex, per-view weight
            w_raw = front_mask * ray_vis.float()                # [B,V,N]
            # Compute Plücker coordinates for each camera-to-point ray            
            cam_pos  = cam_centers.expand(-1, -1, ray_dirs.shape[2], -1)  # [B,2,N,3]
            
            l_vec    = ray_dirs                                         # already norm
            m_vec    = torch.cross(cam_pos, l_vec, dim=-1)              # moment
            plucker  = torch.cat([l_vec, m_vec], dim=-1)                # [B,2,N,6]

            # Build fusion input: latent | plücker | visibility
            vis_feat   = w_raw.unsqueeze(-1)                             # [B,2,N,1]
            fuse_per_view = torch.cat([latent_points, plucker, vis_feat], dim=-1)  # [B,2,N,D+7]

            # ------------------------------------------------------------------
            # Concatenate the two views per point, let the MLP learn the fusion
            # ------------------------------------------------------------------
            fuse_cat = fuse_per_view.permute(0, 2, 1, 3).reshape(
                fuse_per_view.size(0),  # B
                fuse_per_view.size(2),  # N
                -1                      # 2*(D+7)
            )  # [B, N, 2*(D+7)]

            latent_points = self.fusion_mlp(fuse_cat)   # [B, N, D]
            # latent_points_avg = latent_points.mean(dim=1, keepdim=False)
            # latent_points = latent_points_avg + latent_points_fused
            
            # Apply cross-attention with concatenated image features
            # print(f"[DEBUG] image_feats is None: {image_feats is None}")
            # if image_feats is not None:
            #     # Concatenate image features from both source views
            #     # image_feats should be [B, V, S, C] where V=2 for two source views
            #     B, V, S, C = image_feats.shape
            #     concat_img_feats = image_feats.view(B, V * S, C)  # [B, V*S, C] = [B, 2*S, C]
                
            #     # Apply cross-attention between fused latent points and concatenated image features
            #     # print(f"[DEBUG] Applying cross-attention! latent_points: {latent_points.shape}, concat_img_feats: {concat_img_feats.shape}")
            #     latent_points_ca = self.post_fusion_transformer(
            #         latent_points,  # query: [B, N, D]
            #         cond=concat_img_feats,  # key/value: [B, 2*S, C]
            #         mod=None,
            #     )  # [B, N, D]
            #     # skip connection
            #     latent_points = latent_points + latent_points_ca
            #     # print(f"[DEBUG] Cross-attention completed! Output shape: {latent_points.shape}")
            
            # Apply self-attention with layer normalization for stability
            # print(f"[DEBUG] Before self-attention: latent_points range [{latent_points.min():.6f}, {latent_points.max():.6f}], mean: {latent_points.mean():.6f}")
            
            # Pre-normalization for stability
            # latent_points_normed = self.pre_attention_norm(latent_points)  # [B, N, D]
            
            # # MultiheadAttention expects [seq_len, batch, embed_dim] format
            # latent_points_t = latent_points_normed.transpose(0, 1)  # [B, N, D] -> [N, B, D]
            # latent_points_sa_t, _ = self.post_fusion_self_attention(
            #     latent_points_t,  # query: [N, B, D]
            #     latent_points_t,  # key: [N, B, D]  
            #     latent_points_t,  # value: [N, B, D]
            #     need_weights=False  # Don't return attention weights
            # )
            # latent_points_sa = latent_points_sa_t.transpose(0, 1)  # [N, B, D] -> [B, N, D]
            
            # # print(f"[DEBUG] Self-attention output: latent_points_sa range [{latent_points_sa.min():.6f}, {latent_points_sa.max():.6f}], mean: {latent_points_sa.mean():.6f}")
            
            # # Post-normalization and skip connection
            # latent_points_sa_normed = self.post_attention_norm(latent_points_sa)
            # latent_points = latent_points + latent_points_sa_normed
            # print(f"[DEBUG] After skip connection: latent_points range [{latent_points.min():.6f}, {latent_points.max():.6f}], mean: {latent_points.mean():.6f}")

        elif self.num_source_views == 1:
            latent_points = latent_points.squeeze(1)

        render_results = self.renderer(gs_hidden_features=latent_points,
                                       query_points=query_points,
                                       flame_data=flame_params,
                                       w2c=render_w2cs,
                                       intrinsic=render_intrs,
                                       height=render_h,
                                       width=render_w,
                                       background_color=render_bg_colors,
                                       additional_features=None
        )
        

        N, M = render_w2cs.shape[:2]
        assert render_results['comp_rgb'].shape[0] in [N, N], "Batch size mismatch for render_results"
        assert render_results['comp_rgb'].shape[1] in [M, M*2], "Number of rendered views should be consistent with render_cameras"

        return {
            # 'latent_points': latent_points,
            **render_results,
        }
        
    def forward5(self, src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, latent_points, src_canon_2_cam, image_feats=None, source_flame_params=None, render_images=None, data=None):
        assert len(flame_params["betas"].shape) == 2
        render_h, render_w = 512, 512
        query_points = None
        if self.latent_query_points_type.startswith("e2e_flame"):
            query_points, flame_params, surf_normals, faces = self.renderer.get_query_points(flame_params,
                                                                        device=render_w2cs.device)
        if latent_points.ndim >= 3 and self.num_source_views > 1:
            # #### The following part is important to obtain canonical-cam-params #####
            src_w2cs = src_canon_2_cam @ src_w2cs        
            #########################################################################
            R_world2cam = src_w2cs[:, :, :3, :3]           # (B,V,3,3)
            T_world2cam = src_w2cs[:, :, :3, 3]            # (B,V,3)
            # cam_centers = src_w2cs[:, :, :3, 3] # when using cam2world
            ############ Visibility according to vertex normals #####################
            cam_centers = extract_camera_centers(src_w2cs).unsqueeze(2)    # [B, V, 1, 3]
            vertex_positions = query_points.unsqueeze(1)   # [B, 1, N, 3]
            vertex_normals = surf_normals.unsqueeze(1)     # [B, 1, N, 3]
            # rays from vertex toward camera
            ray_dirs = cam_centers - vertex_positions      # [B, V, N, 3]
            ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=-1)
            # front-facing mask
            cos_theta  = (ray_dirs * vertex_normals).sum(-1)   # [B, V, N]
            front_mask = (cos_theta > 0).float() 
            ############### Visibility according to mesh rasterization ##############
            ray_vis = self.vis_mask_rasterizer(   # [B,V,N]  
                        verts=query_points,
                        faces=faces,
                        cam_R=R_world2cam,
                        cam_T=T_world2cam.float(),
                        K=src_intrs)
            # 2. final per-vertex, per-view weight
            w_raw = front_mask * ray_vis.float()                # [B,V,N]
            # Compute Plücker coordinates for each camera-to-point ray            
            cam_pos  = cam_centers.expand(-1, -1, ray_dirs.shape[2], -1)  # [B,2,N,3]
            
            l_vec    = ray_dirs                                         # already norm
            m_vec    = torch.cross(cam_pos, l_vec, dim=-1)              # moment
            plucker  = torch.cat([l_vec, m_vec], dim=-1)                # [B,2,N,6]
            fuse_per_view = torch.cat([latent_points, plucker], dim=-1)  # [B,2,N,D+7]

            # ------------------------------------------------------------------
            # Concatenate the two views per point, let the MLP learn the fusion
            # ------------------------------------------------------------------
            fuse_cat = fuse_per_view.permute(0, 2, 1, 3).reshape(
                fuse_per_view.size(0),  # B
                fuse_per_view.size(2),  # N
                -1                      # 2*(D+7)
            )  # [B, N, 2*(D+7)]

            latent_points_fused = self.fusion_mlp(fuse_cat)   # [B, N, D]
            latent_points_avg = latent_points.mean(dim=1, keepdim=False)
            latent_points = latent_points_avg + latent_points_fused
            
        elif self.num_source_views == 1:
            latent_points = latent_points.squeeze(1)

        render_results = self.renderer(gs_hidden_features=latent_points,
                                       query_points=query_points,
                                       flame_data=flame_params,
                                       w2c=render_w2cs,
                                       intrinsic=render_intrs,
                                       height=render_h,
                                       width=render_w,
                                       background_color=render_bg_colors,
                                       additional_features=None
        )
        

        N, M = render_w2cs.shape[:2]
        assert render_results['comp_rgb'].shape[0] in [N, N], "Batch size mismatch for render_results"
        assert render_results['comp_rgb'].shape[1] in [M, M*2], "Number of rendered views should be consistent with render_cameras"

        return {
            # 'latent_points': latent_points,
            **render_results,
        }
    
    def forward(self, src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, src_canon_2_cam, latent_points=None, image_feats=None, source_flame_params=None, render_images=None, data=None):
        # return self.forward_img_feats(src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, image_feats)
        # return self.forward4(src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, latent_points, image_feats)
        return self.forward4(src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, latent_points, src_canon_2_cam)
        # return self.forward1(src_w2cs, src_intrs, render_w2cs, render_intrs, render_bg_colors, flame_params, latent_points)
        
    @torch.no_grad()
    def infer_single_view(
        self,
        image=None,
        source_c2ws=None,
        source_intrs=None,
        render_w2cs=None,
        render_intrs=None,
        render_bg_colors=None,
        flame_params=None,
        latent_points=None,
    ):
        """
        Render a sequence given one or multiple reference views.

        Two modes are supported:

        1. **End-to-end (default)** – provide `image` with shape
           ``[B, N_ref, C, H, W]``; the method internally runs the encoder and
           transformer to obtain latent point tokens.

        2. **Renderer-only** – provide pre-computed ``latent_points`` with shape
           ``[B, N_ref, N_pts, D]`` (e.g. generated by a separate LAM instance
           that still contained the encoder/transformer).  In this case
           ``image`` can be ``None`` and the current model may have been
           constructed with ``instantiate_encoder=False`` and
           ``instantiate_transformer=False``.
        """

        # ------------------------------------------------------------------ #
        # Validate inputs & determine batch dimensions
        # ------------------------------------------------------------------ #
        if latent_points is None:
            assert image is not None, "Either `image` or `latent_points` must be provided."
            B, N_ref = image.shape[:2]
        else:
            B, N_ref = latent_points.shape[:2]

        assert render_w2cs is not None and render_intrs is not None, \
            "`render_w2cs` and `render_intrs` must be supplied."
        assert render_w2cs.shape[0] == B, "Batch size mismatch for render_w2cs"
        assert render_bg_colors.shape[0] == B, "Batch size mismatch for render_bg_colors"
        assert flame_params["betas"].shape[0] == B, "Batch size mismatch for flame_params"
        assert flame_params["expr"].shape[0] == B, "Batch size mismatch for flame_params"

        latent_points = latent_points.to(render_w2cs)
        source_c2ws = source_c2ws.to(render_w2cs)
        source_intrs = source_intrs.to(render_w2cs)

        # ------------------------------------------------------------------ #
        # Render using the latent points
        # ------------------------------------------------------------------ #
        render_out = self.forward1(
            src_w2cs=source_c2ws,
            src_intrs=source_intrs,
            render_w2cs=render_w2cs,
            render_intrs=render_intrs,
            render_bg_colors=render_bg_colors,
            flame_params=flame_params,
            latent_points=latent_points,
        )

        # Convert to HWC for external consumers
        if "comp_rgb" in render_out and isinstance(render_out["comp_rgb"], torch.Tensor):
            rgb = render_out["comp_rgb"][0].permute(0, 2, 3, 1)  # [Nv, H, W, 3]
            render_out["comp_rgb"] = rgb

        return render_out

    def save_video(self, video_path, image, flame_params, intrinsics, render_bg_color, fps=1, seconds=4, resolution=(512, 512)):
        from tqdm import tqdm

        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        H, W = resolution
        total_frames = seconds * fps

        # Generate circular trajectory
        trajectory = circle_around_axis(
            total_frames,
            axis=Vec3(0, 0, -1),
            up=Vec3(0, 1, 0),
            move=Vec3(0, 0, 1),
            distance=0.3,
        )
        trajectory3 = circle_around_axis(
            total_frames,
            axis=Vec3(0, 1, 0),    # orbiting around Y axis (horizontal)
            up=Vec3(0, 1, 0),
            move=Vec3(0, 0, -1),   # start behind the head
            distance=0.3,
        )

        render_w2cs = torch.stack(
            [torch.from_numpy(np.linalg.inv(p)).float() for p in trajectory + trajectory3], dim=0
        ).unsqueeze(0).to(image.device)

        with VideoWriter(video_path, (W, H), fps=fps) as writer:
            for i in tqdm(range(total_frames*2), desc="Rendering frames"):
                res = self.infer_single_view(
                    image=image,
                    source_c2ws=None,
                    source_intrs=None,
                    render_w2cs=render_w2cs[:, i:i+1],
                    render_intrs=intrinsics,
                    render_bg_colors=render_bg_color,
                    flame_params=flame_params,
                )
                rgb = res['comp_rgb'].cpu().numpy()[0]  # [1, H, W, 3]
                rgb = (rgb * 255).astype(np.uint8)
                writer.add_image(rgb)

        print(f"Video saved to {video_path}")
