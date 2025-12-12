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
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from dataclasses import dataclass, field
from lam.models.rendering.utils.mesh_utils import compute_vertex_normals
from collections import defaultdict
try:
    from diff_gaussian_rasterization_wda import GaussianRasterizationSettings, GaussianRasterizer
except:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from plyfile import PlyData, PlyElement
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import copy
from diffusers.utils import is_torch_version
from lam.models.rendering.flame_model.flame import FlameHeadSubdivided
from lam.models.transformer import TransformerDecoder
from pytorch3d.transforms import matrix_to_quaternion
from lam.models.rendering.utils.typing import *
from lam.models.rendering.utils.utils import trunc_exp, MLP
from lam.models.rendering.gaussian_model import GaussianModel
from einops import rearrange, repeat
from pytorch3d.ops.points_normals import estimate_pointcloud_normals
os.environ["PYOPENGL_PLATFORM"] = "egl"
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.renderer import (
    AmbientLights,
    PerspectiveCameras,
    SoftSilhouetteShader,
    SoftPhongShader,
    RasterizationSettings,
    MeshRenderer,
    MeshRendererWithFragments,
    MeshRasterizer,
    TexturesVertex,
)
from pytorch3d.renderer.blending import BlendParams, softmax_rgb_blend
import lam.models.rendering.utils.mesh_utils as mesh_utils
from lam.models.rendering.utils.point_utils import depth_to_normal
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes

def visualize_gaussians_as_ellipsoids(
    gs_model, 
    save_path: str,
    n_rings: int = 8,
    n_sectors: int = 8,
    scale_factor: float = 1.0,
    opacity_threshold: float = 0.1,
    max_gaussians: int = None,
):
    """
    Visualize Gaussian primitives as ellipsoid meshes and save to OBJ file.
    
    Args:
        gs_model: GaussianModel with xyz, rotation (quaternion), scaling, opacity, shs
        save_path: Output path for OBJ file
        n_rings: Number of latitude rings for ellipsoid mesh
        n_sectors: Number of longitude sectors for ellipsoid mesh  
        scale_factor: Additional scaling factor for ellipsoids
        opacity_threshold: Only visualize Gaussians with opacity above this threshold
        max_gaussians: If set, limit to this many Gaussians (for performance)
    """
    import os
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    
    # Extract Gaussian attributes
    xyz = gs_model.xyz.detach().cpu().numpy()  # [N, 3]
    rotation = gs_model.rotation.detach().cpu().numpy()  # [N, 4] quaternion (w, x, y, z)
    scaling = gs_model.scaling.detach().cpu().numpy()  # [N, 3]
    opacity = gs_model.opacity.detach().cpu().numpy()  # [N, 1]
    shs = gs_model.shs.detach().cpu().numpy()  # [N, K, 3] or [N, 3]
    
    # Get colors from SH DC term
    if len(shs.shape) == 3:
        colors = shs[:, 0, :]  # [N, 3]
    else:
        colors = shs[:, :3]  # [N, 3]
    colors = np.clip(colors, 0, 1)
    
    # Filter by opacity
    mask = opacity.squeeze() > opacity_threshold
    xyz = xyz[mask]
    rotation = rotation[mask]
    scaling = scaling[mask]
    colors = colors[mask]
    
    # Limit number of Gaussians if specified
    if max_gaussians is not None and len(xyz) > max_gaussians:
        indices = np.random.choice(len(xyz), max_gaussians, replace=False)
        xyz = xyz[indices]
        rotation = rotation[indices]
        scaling = scaling[indices]
        colors = colors[indices]
    
    print(f"[visualize_gaussians] Visualizing {len(xyz)} Gaussians as ellipsoids")
    
    # Generate unit sphere vertices and faces
    sphere_verts = []
    for i in range(n_rings + 1):
        phi = np.pi * i / n_rings
        for j in range(n_sectors):
            theta = 2 * np.pi * j / n_sectors
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            sphere_verts.append([x, y, z])
    sphere_verts = np.array(sphere_verts)
    
    sphere_faces = []
    for i in range(n_rings):
        for j in range(n_sectors):
            v0 = i * n_sectors + j
            v1 = i * n_sectors + (j + 1) % n_sectors
            v2 = (i + 1) * n_sectors + j
            v3 = (i + 1) * n_sectors + (j + 1) % n_sectors
            sphere_faces.append([v0, v2, v1])
            sphere_faces.append([v1, v2, v3])
    sphere_faces = np.array(sphere_faces)
    
    n_verts_per_ellipsoid = len(sphere_verts)
    n_faces_per_ellipsoid = len(sphere_faces)
    
    # Quaternion to rotation matrix (w, x, y, z convention)
    def quat_to_rotmat(q):
        w, x, y, z = q[0], q[1], q[2], q[3]
        R = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
            [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
        ])
        return R
    
    all_verts = []
    all_faces = []
    all_colors = []
    
    for i in range(len(xyz)):
        # Get rotation matrix from quaternion
        R = quat_to_rotmat(rotation[i])
        
        # Scale sphere to ellipsoid
        S = np.diag(scaling[i] * scale_factor)
        
        # Transform vertices: first scale, then rotate, then translate
        transformed_verts = (R @ S @ sphere_verts.T).T + xyz[i]
        
        # Offset face indices
        offset_faces = sphere_faces + i * n_verts_per_ellipsoid
        
        all_verts.append(transformed_verts)
        all_faces.append(offset_faces)
        all_colors.extend([colors[i]] * n_verts_per_ellipsoid)
    
    all_verts = np.vstack(all_verts)
    all_faces = np.vstack(all_faces)
    all_colors = np.array(all_colors)
    
    # Save as OBJ with vertex colors (using material)
    mtl_path = save_path.replace('.obj', '.mtl')
    mtl_name = os.path.basename(mtl_path)
    
    with open(save_path, 'w') as f:
        f.write(f"# Gaussian Ellipsoids Visualization\n")
        f.write(f"# {len(xyz)} Gaussians, {len(all_verts)} vertices, {len(all_faces)} faces\n")
        f.write(f"mtllib {mtl_name}\n")
        
        # Write vertices with colors (OBJ extension: v x y z r g b)
        for v, c in zip(all_verts, all_colors):
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
        
        # Write faces (1-indexed)
        for face in all_faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")
    
    # Also save as PLY for better color support
    ply_path = save_path.replace('.obj', '.ply')
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(all_verts)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write(f"element face {len(all_faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        
        for v, c in zip(all_verts, all_colors):
            r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f} {r} {g} {b}\n")
        
        for face in all_faces:
            f.write(f"3 {face[0]} {face[1]} {face[2]}\n")
    
    print(f"[visualize_gaussians] Saved ellipsoid mesh to: {save_path}")
    print(f"[visualize_gaussians] Saved PLY with colors to: {ply_path}")
    
    return save_path, ply_path


inverse_sigmoid = lambda x: np.log(x / (1 - x))


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY, device):
    tanHalfFovY = torch.tan((fovY / 2))
    tanHalfFovX = torch.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right
    P = torch.zeros(4, 4, device=device)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def intrinsic_to_fov(intrinsic, w, h):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    fov_x = 2 * torch.arctan2(w, 2 * fx)
    fov_y = 2 * torch.arctan2(h, 2 * fy)
    return fov_x, fov_y

 
class Camera:
    def __init__(self, w2c, intrinsic, FoVx, FoVy, height, width, trans=np.array([0.0, 0.0, 0.0]), scale=1.0) -> None:
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.height = int(height)
        self.width = int(width)
        self.world_view_transform = w2c.transpose(0, 1)
        self.intrinsic = intrinsic

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, device=w2c.device).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # overkill
        # self.camera_center = self.world_view_transform.inverse()[3, :3]
        R = w2c[..., :3, :3]
        t = w2c[..., :3, 3]
        self.camera_center = -(R.transpose(-1, -2) @ t.unsqueeze(-1)).squeeze(-1).float()

    @staticmethod
    # @torch.compile
    def from_c2w(w2c, intrinsic, height, width):
        # w2c = torch.inverse(c2w)
        FoVx, FoVy = intrinsic_to_fov(intrinsic, w=torch.tensor(width, device=w2c.device), h=torch.tensor(height, device=w2c.device))
        return Camera(w2c=w2c, intrinsic=intrinsic, FoVx=FoVx, FoVy=FoVy, height=height, width=width)


class GSLayer(nn.Module):
    def __init__(self, in_channels, use_rgb, 
                 clip_scaling=0.2, 
                 init_scaling=-5.0,
                 scale_sphere=False,
                 init_density=0.1,
                 sh_degree=None, 
                 xyz_offset=True,
                 restrict_offset=True,
                 xyz_offset_max_step=None,
                 fix_opacity=False,
                 fix_rotation=False,
                 use_fine_feat=False,
                 pred_res=False,
                 ):
        super().__init__()
        self.clip_scaling = clip_scaling
        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.use_fine_feat = use_fine_feat
        self.scale_sphere = scale_sphere
        self.pred_res = pred_res
        
        self.attr_dict ={
            "shs": (sh_degree + 1) ** 2 * 3,
            "scaling": 3 if not scale_sphere else 1,
            "xyz": 3,
            "opacity": None,
            "rotation": None 
        }
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1
        if not self.fix_rotation:
            self.attr_dict["rotation"] = 4
        
        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs" and use_rgb:
                    out_ch = 3
                if key == "shs":
                    shs_out_ch = out_ch
                if pred_res:
                    layer = nn.Linear(in_channels+out_ch, out_ch)
                else:
                    layer = nn.Linear(in_channels, out_ch)
            # initialize
            if not (key == "shs" and use_rgb):
                if key == "opacity" and self.fix_opacity:
                    pass
                elif key == "rotation" and self.fix_rotation:
                    pass
                else:
                    nn.init.constant_(layer.weight, 0)
                    nn.init.constant_(layer.bias, 0)
            if key == "scaling":
                nn.init.constant_(layer.bias, init_scaling)
            elif key == "rotation":
                if not self.fix_rotation:
                    nn.init.constant_(layer.bias, 0)
                    nn.init.constant_(layer.bias[0], 1.0)
            elif key == "opacity":
                if not self.fix_opacity:
                    nn.init.constant_(layer.bias, inverse_sigmoid(init_density))
            self.out_layers[key] = layer
            
        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer
            
    def forward(self, x, pts, x_fine=None, gs_raw_attr=None, ret_raw=False, vtx_sym_idxs=None):
        assert len(x.shape) == 2
        ret = {}
        if ret_raw:
            raw_attr = {}
        ori_x = x
        for k in self.attr_dict:
            # if vtx_sym_idxs is not None and k in ["shs", "scaling", "opacity"]:
            if vtx_sym_idxs is not None and k in ["shs", "scaling", "opacity", "rotation"]:
                # print("==="*16*3, "\n\n\n"+"use sym mean.", "\n"+"==="*16*3)
                # x = (x + x[vtx_sym_idxs.to(x.device), :]) / 2.
                x = ori_x[vtx_sym_idxs.to(x.device), :]
            else:
                x = ori_x
            layer =self.out_layers[k]
            if self.pred_res and (not self.fix_opacity or k != "opacity") and (not self.fix_rotation or k != "rotation"):
                v = layer(torch.cat([gs_raw_attr[k], x], dim=-1))
                v = gs_raw_attr[k] + v
            else:
                v = layer(x)
            if ret_raw:
                raw_attr[k] = v 
            if k == "rotation":
                if self.fix_rotation:
                    v = matrix_to_quaternion(torch.eye(3).type_as(x)[None,: , :].repeat(x.shape[0], 1, 1)) # constant rotation
                else:
                    # assert len(x.shape) == 2
                    v = torch.nn.functional.normalize(v)
            elif k == "scaling":
                v = trunc_exp(v)
                if self.scale_sphere:
                    assert v.shape[-1] == 1
                    v = torch.cat([v, v, v], dim=-1)
                if self.clip_scaling is not None:
                    v = torch.clamp(v, min=0, max=self.clip_scaling)
            elif k == "opacity":
                if self.fix_opacity:
                    v = torch.ones_like(x)[..., 0:1]
                else:
                    v = torch.sigmoid(v)
            elif k == "shs":
                if self.use_rgb:
                    v[..., :3] = torch.sigmoid(v[..., :3])
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v_fine = torch.tanh(v_fine)
                        v = v + v_fine
                else:
                    if self.use_fine_feat:
                        v_fine = self.out_layers["fine_shs"](x_fine)
                        v = v + v_fine
                v = torch.reshape(v, (v.shape[0], -1, 3))
            elif k == "xyz":
                # TODO check
                if self.restrict_offset:
                    max_step = self.xyz_offset_max_step
                    v = (torch.sigmoid(v) - 0.5) * max_step
                if self.xyz_offset:
                    pass
                else:
                    assert NotImplementedError
                ret["offset"] = v
                v = pts + v
            ret[k] = v
            
        if ret_raw:
            return GaussianModel(**ret), raw_attr
        else:
            return GaussianModel(**ret)


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48, dim=128):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack([
            torch.cat([e, torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6), e,
                        torch.zeros(self.embedding_dim // 6)]),
            torch.cat([torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6), e]),
        ])
        self.register_buffer('basis', e)  # 3 x 16

        self.mlp = nn.Linear(self.embedding_dim+3, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum(
            'bnd,de->bne', input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)
        return embeddings
    
    def forward(self, input):
        # input: B x N x 3
        embed = self.mlp(torch.cat([self.embed(input, self.basis), input], dim=2)) # B x N x C
        embed = self.norm(embed)
        return embed


class CrossAttnBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """
    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float=None,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0., feedforward=False):
        super().__init__()
        # TODO check already apply normalization
        # self.norm_q = nn.LayerNorm(inner_dim, eps=eps)
        # self.norm_k = nn.LayerNorm(cond_dim, eps=eps)
        self.norm_q = nn.Identity()
        self.norm_k = nn.Identity()
        
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        
        self.mlp = None
        if feedforward:
            self.norm2 = nn.LayerNorm(inner_dim, eps=eps)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=inner_dim, num_heads=num_heads,
                dropout=attn_drop, bias=attn_bias, batch_first=True)
            self.norm3 = nn.LayerNorm(inner_dim, eps=eps)
            self.mlp = nn.Sequential(
                nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
                nn.GELU(),
                nn.Dropout(mlp_drop),
                nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
                nn.Dropout(mlp_drop),
            )

    def forward(self, x, cond):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        x = self.cross_attn(self.norm_q(x), self.norm_k(cond), cond, need_weights=False)[0]
        if self.mlp is not None:
            before_sa = self.norm2(x)
            x = x + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
            x = x + self.mlp(self.norm3(x))
        return x
    
    
class DecoderCrossAttn(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads, mlp=False, decode_with_extra_info=None):
        super().__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.cross_attn = CrossAttnBlock(inner_dim=query_dim, cond_dim=context_dim, 
                                         num_heads=num_heads, feedforward=mlp,
                                         eps=1e-5)
        self.decode_with_extra_info = decode_with_extra_info
        if decode_with_extra_info is not None:
            if decode_with_extra_info["type"] == "dinov2p14_feat":
                context_dim = decode_with_extra_info["cond_dim"]
                self.cross_attn_color = CrossAttnBlock(inner_dim=query_dim, cond_dim=context_dim, 
                                            num_heads=num_heads, feedforward=False, eps=1e-5)
            elif decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                from lam.models.encoders.dinov2_wrapper import Dinov2Wrapper
                self.encoder = Dinov2Wrapper(model_name='dinov2_vits14_reg', freeze=False, encoder_feat_dim=384)
                self.cross_attn_color = CrossAttnBlock(inner_dim=query_dim, cond_dim=384, 
                                            num_heads=num_heads, feedforward=False,
                                            eps=1e-5)
            elif decode_with_extra_info["type"] == "decoder_resnet18_feat":
                from lam.models.encoders.xunet_wrapper import XnetWrapper
                self.encoder = XnetWrapper(model_name='resnet18', freeze=False, encoder_feat_dim=64)
                self.cross_attn_color = CrossAttnBlock(inner_dim=query_dim, cond_dim=64, 
                                            num_heads=num_heads, feedforward=False,
                                            eps=1e-5)
                
    def resize_image(self, image, multiply):
        B, _, H, W = image.shape
        new_h, new_w = math.ceil(H / multiply) * multiply, math.ceil(W / multiply) * multiply
        image = F.interpolate(image, (new_h, new_w), align_corners=True, mode="bilinear")
        return image
    
    def forward(self, pcl_query,  pcl_latent, extra_info=None):
        out = self.cross_attn(pcl_query, pcl_latent)
        if self.decode_with_extra_info is not None:
            out_dict = {}
            out_dict["coarse"] = out
            if self.decode_with_extra_info["type"] == "dinov2p14_feat":
                out = self.cross_attn_color(out, extra_info["image_feats"])
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_dinov2p14_feat":
                img_feat = self.encoder(extra_info["image"])
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
            elif self.decode_with_extra_info["type"] == "decoder_resnet18_feat":
                image = extra_info["image"]
                image = self.resize_image(image, multiply=32)
                img_feat = self.encoder(image)
                out = self.cross_attn_color(out, img_feat)
                out_dict["fine"] = out
                return out_dict
        return out


class GS3DRenderer(nn.Module):
    def __init__(self, human_model_path, subdivide_num, smpl_type, feat_dim, query_dim, 
                 use_rgb, sh_degree, xyz_offset_max_step, mlp_network_config,
                 expr_param_dim, shape_param_dim,
                 clip_scaling=0.2,
                 scale_sphere=False,
                 skip_decoder=False,
                 fix_opacity=False,
                 fix_rotation=False,
                 decode_with_extra_info=None,
                 gradient_checkpointing=False,
                 add_teeth=True,
                 teeth_bs_flag=False,
                 oral_mesh_flag=False,
                 num_gaussians_per_vertex=1,
                 is_finetuning=False,
                 **kwargs,
                 ):
        super().__init__()
        print(f"#########scale sphere:{scale_sphere}, add_teeth:{add_teeth}")
        self.gradient_checkpointing = gradient_checkpointing
        self.skip_decoder = skip_decoder
        self.smpl_type = smpl_type
        assert self.smpl_type == "flame"
        self.sym_rend2 = True
        self.teeth_bs_flag = teeth_bs_flag
        self.oral_mesh_flag = oral_mesh_flag
        self.render_rgb = kwargs.get("render_rgb", True)
        self.num_gaussians_per_vertex = num_gaussians_per_vertex
        self.is_finetuning = is_finetuning
        print("==="*16*3, f"\n Render rgb: {self.render_rgb}, Gaussians per vertex: {self.num_gaussians_per_vertex} \n"+"==="*16*3)
        
        self.scaling_modifier = 1.0
        self.sh_degree = sh_degree
        if use_rgb:
            self.sh_degree = 0

        use_rgb = use_rgb

        self.flame_model = FlameHeadSubdivided(
            300,
            100,
            add_teeth=add_teeth,
            add_shoulder=False,
            flame_model_path=f'{human_model_path}/flame_assets/flame/flame2020.pkl',
            flame_lmk_embedding_path=f"{human_model_path}/flame_assets/flame/landmark_embedding_with_eyes.npy",
            flame_template_mesh_path=f"{human_model_path}/flame_assets/flame/head_template_mesh.obj",
            flame_parts_path=f"{human_model_path}/flame_assets/flame/FLAME_masks.pkl",
            subdivide_num=subdivide_num,
            teeth_bs_flag=teeth_bs_flag,
            oral_mesh_flag=oral_mesh_flag
        )

        if not self.skip_decoder:
            self.pcl_embed = PointEmbed(dim=query_dim)

        self.mlp_network_config = mlp_network_config
        if self.mlp_network_config is not None:
            # Always create MLPs as a ModuleList for consistency
            self.mlp_nets = nn.ModuleList()
            
            # First MLP: original (preserves learned weights)
            self.mlp_net = MLP(query_dim, query_dim, **self.mlp_network_config)

        init_scaling = -7.0  # Smaller initial Gaussians: exp(-7) ≈ 0.0009 vs exp(-5) ≈ 0.0067
        # Single gs_net - same for all gaussians (just converts features to attributes)
        self.gs_net = GSLayer(in_channels=query_dim,
                              use_rgb=use_rgb,
                              sh_degree=self.sh_degree,
                              clip_scaling=clip_scaling,
                              scale_sphere=scale_sphere,
                              init_scaling=init_scaling,
                              init_density=0.1,
                              xyz_offset=True,
                              restrict_offset=True,
                              xyz_offset_max_step=xyz_offset_max_step,
                              fix_opacity=fix_opacity,
                              fix_rotation=fix_rotation,
                              use_fine_feat=True if decode_with_extra_info is not None and decode_with_extra_info["type"] is not None else False,
                              )
        
    def build_mlps(self):
        if self.num_gaussians_per_vertex != 1:
            # Additional MLPs: perturbed copies
            self.mlp_nets.append(self.mlp_net)
            for i in range(1, self.num_gaussians_per_vertex):
                perturbed_mlp = self._create_perturbed_mlp(self.mlp_nets[0])
                self.mlp_nets.append(perturbed_mlp)

    def _create_perturbed_mlp(self, base_mlp):
        """Create a perturbed copy of the MLP with small weight variations"""
        import copy
        perturbed_mlp = copy.deepcopy(base_mlp)

        noise_scale = 0.0001
        
        with torch.no_grad():
            for name, param in perturbed_mlp.named_parameters():
                if 'weight' in name or 'bias' in name:
                    noise = torch.randn_like(param) * noise_scale
                    param.add_(noise)
        
        return perturbed_mlp
        
    def forward_single_view(self,
        gs: GaussianModel,
        viewpoint_camera: Camera,
        background_color: Optional[Float[Tensor, "3"]],
        ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(gs.xyz, dtype=gs.xyz.dtype, requires_grad=True, device=self.device) + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        
        bg_color = background_color
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        GSRSettings = GaussianRasterizationSettings
        GSR = GaussianRasterizer

        raster_settings = GSRSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=self.scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform.float(),
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GSR(raster_settings=raster_settings)

        means3D = gs.xyz
        means2D = screenspace_points
        opacity = gs.opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        scales = gs.scaling
        rotations = gs.rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if self.gs_net.use_rgb:
            colors_precomp = gs.shs.squeeze(1)
        else:
            shs = gs.shs
        # Rasterize visible Gaussians to image, obtain their radii (on screen). 
        # torch.cuda.synchronize()
        # with boxx.timeit():
        with torch.autocast(device_type=self.device.type, dtype=torch.float32):
            raster_ret = rasterizer(
                means3D = means3D.float(),
                means2D = means2D.float(),
                shs = shs.float() if not self.gs_net.use_rgb else None,
                colors_precomp = colors_precomp.float() if colors_precomp is not None else None,
                opacities = opacity.float(),
                scales = scales.float(),
                rotations = rotations.float(),
                cov3D_precomp = cov3D_precomp
            )
        rendered_image, radii, rendered_depth, rendered_alpha = raster_ret

        ret = {
            "comp_rgb": rendered_image.permute(1, 2, 0),  # [H, W, 3]
            "comp_rgb_bg": bg_color,
            'comp_mask': rendered_alpha.permute(1, 2, 0),
            'comp_depth': rendered_depth.permute(1, 2, 0),
        }

        return ret
    
    def animate_gs_model_finetuning(self, gs_attr: GaussianModel, flame_data):
        num_view = flame_data["expr"].shape[0]
        num_gaussians_per_vertex = getattr(gs_attr, 'num_gaussians_per_vertex', 1)

        expr = flame_data["expr"]
        shape = flame_data["betas"].repeat(num_view, 1)

        # compute transforms once from base mesh
        A = self.flame_model.get_cano_transform(
            shape=shape,
            expr=expr,
            rotation=flame_data["rotation"],
            neck=flame_data["neck_pose"],
            jaw=flame_data["jaw_pose"],
            eyes=flame_data["eyes_pose"],
            batch_size=shape.shape[0],
        )

        # canonical gaussians for all (V*K,3)
        mean_3d = gs_attr.xyz.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, V*K, 3]

        ret = self.flame_model.animation_forward_finetuning(
            v_cano=mean_3d,
            expr=expr,
            translation=flame_data["translation"],
            batch_size=shape.shape[0],
            A=A,
            num_gaussians_per_vertex=num_gaussians_per_vertex,
        )
        mean_3d = ret["animated"]  # [Nv, V*K, 3]

        # build per-view GaussianModels
        gs_attr_list = []
        for i in range(num_view):
            gs_attr_copy = GaussianModel(
                xyz=mean_3d[i],
                opacity=gs_attr.opacity,
                rotation=gs_attr.rotation,
                scaling=gs_attr.scaling,
                shs=gs_attr.shs,
                offset=gs_attr.offset,
            )
            gs_attr_list.append(gs_attr_copy)

        return gs_attr_list
            
    def animate_gs_model(self, gs_attr: GaussianModel, query_points, flame_data, debug=False):
        """
        query_points: [N, 3]
        """
        device = gs_attr.xyz.device

        with torch.autocast(device_type=device.type, dtype=torch.float32):
            # mean_3d = query_points + gs_attr.xyz  # [N, 3]
            mean_3d = gs_attr.xyz  # [N, 3]
            
            num_view = flame_data["expr"].shape[0]  # [Nv, 100]
            mean_3d = mean_3d.unsqueeze(0).repeat(num_view, 1, 1)  # [Nv, N, 3]
            query_points = query_points.unsqueeze(0).repeat(num_view, 1, 1)

            expr = flame_data["expr"]
            ret = self.flame_model.animation_forward(v_cano=mean_3d,
                                                shape=flame_data["betas"].repeat(num_view, 1),
                                                expr=expr,
                                                rotation=flame_data["rotation"],
                                                neck=flame_data["neck_pose"],
                                                jaw=flame_data["jaw_pose"],
                                                eyes=flame_data["eyes_pose"],
                                                translation=flame_data["translation"],
                                                zero_centered_at_root_node=False,
                                                return_landmarks=False,
                                                return_verts_cano=False,
                                                # static_offset=flame_data['static_offset'].to('cuda'),
                                                static_offset=None,
                                                )
            mean_3d = ret["animated"]
            
        gs_attr_list = []                                                                  
        for i in range(num_view):
            gs_attr_copy = GaussianModel(xyz=mean_3d[i],
                                    opacity=gs_attr.opacity, 
                                    rotation=gs_attr.rotation, 
                                    scaling=gs_attr.scaling,
                                    shs=gs_attr.shs,
                                    offset=gs_attr.offset) # [N, 3]
            gs_attr_list.append(gs_attr_copy)
        
        return gs_attr_list
        
    
    def forward_gs_attr(self, x, query_points, flame_data, debug=False, x_fine=None, vtx_sym_idxs=None):
        """
        x: [V, C] - latent features from cross-attention 
        query_points: [V, 3] - FLAME vertex positions
        """
        device = x.device
        
        # Original behavior when num_gaussians_per_vertex == 1
        if self.num_gaussians_per_vertex == 1:
            if self.mlp_network_config is not None:
                x = self.mlp_net(x)
                if x_fine is not None:
                    x_fine = self.mlp_net(x_fine)
            gs_attr = self.gs_net(x, query_points, x_fine, vtx_sym_idxs=vtx_sym_idxs)
            return gs_attr
        
        # Multi-gaussian behavior: different MLPs process same input → different features → different gaussians
        all_gs_attrs = []
        
        for i in range(self.num_gaussians_per_vertex):
            # Process features with i-th MLP
            if self.mlp_network_config is not None:
                x_processed = self.mlp_nets[i](x)  # Different MLP = different processing
                x_fine_processed = self.mlp_nets[i](x_fine) if x_fine is not None else None
            else:
                x_processed = x
                x_fine_processed = x_fine
            
            # Same gs_net converts different input features to different gaussian attributes
            gs_attr_i = self.gs_net(x_processed, query_points, x_fine_processed, vtx_sym_idxs=vtx_sym_idxs)
            all_gs_attrs.append(gs_attr_i)
        
        # Concatenate all gaussians
        combined_attrs = {}
        for attr_name in ['xyz', 'opacity', 'rotation', 'scaling', 'shs', 'offset']:
            if hasattr(all_gs_attrs[0], attr_name) and getattr(all_gs_attrs[0], attr_name) is not None:
                combined_attrs[attr_name] = torch.cat([getattr(gs, attr_name) for gs in all_gs_attrs], dim=0)
            else:
                combined_attrs[attr_name] = None
        combined_gs = GaussianModel(**combined_attrs)
        combined_gs.num_gaussians_per_vertex = self.num_gaussians_per_vertex
        return combined_gs
            

    def get_query_points(self, flame_data, device):
        with torch.no_grad():
            with torch.autocast(device_type=device.type, dtype=torch.float32):
                # print(flame_data["betas"].shape, flame_data["face_offset"].shape, flame_data["joint_offset"].shape)
                # positions, _, transform_mat_neutral_pose = self.flame_model.get_query_points(flame_data, device=device)  # [B, N, 3]
                positions = self.flame_model.get_cano_verts(shape_params=flame_data["betas"])  # [B, N, 3]
                mesh = Meshes(positions, self.flame_model.faces_up.expand(positions.shape[0], -1, -1))
                
                # # Save mesh(es) to OBJ file
                # import trimesh
                # os.makedirs("debug_meshes", exist_ok=True)
                # batch_size = positions.shape[0]
                # for b in range(batch_size):
                #     verts_np = positions[b].detach().cpu().numpy()  # [N, 3]
                #     faces_np = self.flame_model.faces_up.detach().cpu().numpy()  # [F, 3]
                    
                #     # Create trimesh object and save
                #     mesh_obj = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
                #     mesh_obj.export(f"debug_meshes/flame_canonical_batch{b}.obj")
                #     print(f"Saved mesh to debug_meshes/flame_canonical_batch{b}.obj")
                
                # print(f"positions shape:{positions.shape}")
                
        return positions, flame_data, mesh.verts_normals_packed().view_as(positions), self.flame_model.faces_up
    
    def query_latent_feat(self,
                          positions: Float[Tensor, "*B N1 3"],
                          flame_data,
                          latent_feat: Float[Tensor, "*B N2 C"],
                          extra_info):
        device = latent_feat.device
        if self.skip_decoder:
            gs_feats = latent_feat
            assert positions is not None
        else:
            assert positions is None
            if positions is None:
                positions, flame_data = self.get_query_points(flame_data, device)

            with torch.autocast(device_type=device.type, dtype=torch.float32):
                pcl_embed = self.pcl_embed(positions)
                gs_feats = pcl_embed

        return gs_feats, positions, flame_data

    def forward_single_batch(
        self,
        gs_list: list[GaussianModel],
        w2cs: Float[Tensor, "Nv 4 4"],
        intrinsics: Float[Tensor, "Nv 4 4"],
        height: int,
        width: int,
        background_color: Optional[Float[Tensor, "Nv 3"]],
        debug: bool=False,
    ):
        out_list = []
        self.device = gs_list[0].xyz.device
        for v_idx, (w2c, intrinsic) in enumerate(zip(w2cs, intrinsics)):
            out_list.append(self.forward_single_view(
                                gs_list[v_idx], 
                                Camera.from_c2w(w2c, intrinsic, height, width),
                                background_color[v_idx]
                            ))
        
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        out = {k: torch.stack(v, dim=0) for k, v in out.items()}
        out["3dgs"] = gs_list
        offsets = [gs.offset if gs.offset is not None else torch.zeros_like(gs.xyz)
                for gs in gs_list]

        out["offset"] = torch.stack(offsets, dim=0)

        return out

    def get_sing_batch_smpl_data(self, smpl_data, bidx):
        smpl_data_single_batch = {}
        for k, v in smpl_data.items():
            smpl_data_single_batch[k] = v[bidx]  # e.g. body_pose: [B, N_v, 21, 3] -> [N_v, 21, 3]
            if k == "betas" or (k == "joint_offset") or (k == "face_offset"):
                smpl_data_single_batch[k] = v[bidx:bidx+1]  # e.g. betas: [B, 100] -> [1, 100]
        return smpl_data_single_batch
    
    def get_single_view_smpl_data(self, smpl_data, vidx):
        smpl_data_single_view = {}        
        for k, v in smpl_data.items():
            assert v.shape[0] == 1
            if k == "betas" or (k == "joint_offset") or (k == "face_offset") or (k == "transform_mat_neutral_pose"):
                smpl_data_single_view[k] = v  # e.g. betas: [1, 100] -> [1, 100]
            else:
                smpl_data_single_view[k] = v[:, vidx: vidx + 1]  # e.g. body_pose: [1, N_v, 21, 3] -> [1, 1, 21, 3]
        return smpl_data_single_view
            
    def forward_gs(self, 
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np_q 3"],
        flame_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        additional_features: Optional[dict] = None,
        debug: bool = False,
        **kwargs):
                
        batch_size = gs_hidden_features.shape[0]
        
        query_gs_features, query_points, flame_data = self.query_latent_feat(query_points, flame_data, gs_hidden_features,
                                                                             additional_features)

        gs_model_list = []
        all_query_points = []
        for b in range(batch_size):
            all_query_points.append(query_points[b:b+1, :])
            if isinstance(query_gs_features, dict):
                ret_gs = self.forward_gs_attr(query_gs_features["coarse"][b], query_points[b], None, debug, 
                                                x_fine=query_gs_features["fine"][b], vtx_sym_idxs=None)
            else:
                ret_gs = self.forward_gs_attr(query_gs_features[b], query_points[b], None, debug, vtx_sym_idxs=None)

            gs_model_list.append(ret_gs)

        query_points = torch.cat(all_query_points, dim=0)
        return gs_model_list, query_points, flame_data, query_gs_features

    def forward_res_refine_gs(self, 
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np_q 3"],
        flame_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        additional_features: Optional[dict] = None,
        debug: bool = False,
        gs_raw_attr_list: list = None,
        **kwargs):
                
        batch_size = gs_hidden_features.shape[0]
        
        query_gs_features, query_points, flame_data = self.query_latent_feat(query_points, flame_data, gs_hidden_features,
                                                                             additional_features)

        gs_model_list = []
        for b in range(batch_size):
            gs_model = self.gs_refine_net(query_gs_features[b], query_points[b], x_fine=None, gs_raw_attr=gs_raw_attr_list[b])
            gs_model_list.append(gs_model)
        return gs_model_list, query_points, flame_data, query_gs_features

    def forward_animate_gs(self, gs_model_list, query_points, flame_data, w2c, intrinsic, height, width,
                           background_color, debug=False):
        batch_size = len(gs_model_list)
        out_list = []

        for b in range(batch_size):
            gs_model = gs_model_list[b]
            query_pt = query_points[b]
            # with torch.autograd.profiler.record_function("animate_gs_model"):
            if self.is_finetuning:
                animatable_gs_model_list: list[GaussianModel] = self.animate_gs_model_finetuning(gs_model,
                                                                                                self.get_sing_batch_smpl_data(flame_data, b))
            else:
                animatable_gs_model_list: list[GaussianModel] = self.animate_gs_model(gs_model,
                                                                                  query_pt,
                                                                                  self.get_sing_batch_smpl_data(flame_data, b),
                                                                                  debug=debug)
            assert len(animatable_gs_model_list) == w2c.shape[1]
            out_list.append(self.forward_single_batch(
                animatable_gs_model_list,
                w2c[b],
                intrinsic[b],
                height, width,
                background_color[b] if background_color is not None else None, 
                debug=debug))
            
        out = defaultdict(list)
        for out_ in out_list:
            for k, v in out_.items():
                out[k].append(v)
        for k, v in out.items():
            if isinstance(v[0], torch.Tensor):
                out[k] = torch.stack(v, dim=0)
            else:
                out[k] = v
                
        render_keys = ["comp_rgb", "comp_mask"] # , "comp_depth"
        for key in render_keys:
            out[key] = rearrange(out[key], "b v h w c -> b v c h w")
        
        return out

    def project_single_view_feats(self, img_vtx_ids, feats, nv, inter_feat=True):
        b, h, w, k = img_vtx_ids.shape
        c, ih, iw = feats.shape
        vtx_ids = img_vtx_ids
        if h != ih or w != iw:
            if inter_feat:
                feats = torch.nn.functional.interpolate(
                    rearrange(feats, "(b c) h w -> b c h w", b=1).float(), (h, w)
                ).squeeze(0)
                vtx_ids = rearrange(vtx_ids, "b (c h) w k -> (b k) c h w", c=1).long().squeeze(1)
            else:
                vtx_ids = torch.nn.functional.interpolate(
                    rearrange(vtx_ids, "b (c h) w k -> (b k) c h w", c=1).float(), (ih, iw), mode="nearest"
                ).long().squeeze(1)
        else:
            vtx_ids = rearrange(vtx_ids, "b h w k -> (b k) h w", b=1).long()
        vis_mask = vtx_ids > 0
        vtx_ids = vtx_ids[vis_mask]  # n
        vtx_ids = repeat(vtx_ids, "n -> n c", c=c)

        feats = repeat(feats, "c h w -> k h w c", k=k).to(vtx_ids.device)
        feats = feats[vis_mask, :] # n, c

        sums = torch.zeros((nv, c), dtype=feats.dtype, device=feats.device)
        counts = torch.zeros((nv), dtype=torch.int64, device=feats.device)

        sums.scatter_add_(0, vtx_ids, feats)
        one_hot = torch.ones_like(vtx_ids[:, 0], dtype=torch.int64).to(feats.device)
        counts.scatter_add_(0, vtx_ids[:, 0], one_hot)
        clamp_counts = counts.clamp(min=1)
        mean_feats = sums / clamp_counts.view(-1, 1) 
        return mean_feats
    
    def forward(self, 
        gs_hidden_features: Float[Tensor, "B Np Cp"],
        query_points: Float[Tensor, "B Np 3"],
        flame_data,  # e.g., body_pose:[B, Nv, 21, 3], betas:[B, 100]
        w2c: Float[Tensor, "B Nv 4 4"],
        intrinsic: Float[Tensor, "B Nv 4 4"],
        height,
        width,
        additional_features: Optional[Float[Tensor, "B C H W"]] = None,
        background_color: Optional[Float[Tensor, "B Nv 3"]] = None,
        debug: bool = False,
        **kwargs):
        
        # need shape_params of flame_data to get querty points and get "transform_mat_neutral_pose"
        # with torch.autograd.profiler.record_function("forward_gs"):
        gs_model_list, query_points, flame_data, query_gs_features = self.forward_gs(gs_hidden_features, query_points, flame_data=flame_data,
                                                                      additional_features=additional_features, debug=debug)
        # breakpoint()
        # # if os.environ.get("VIS_CANONICAL_GAUSSIANS", "0") == "1":
        # vis_dir = os.environ.get("VIS_CANONICAL_GAUSSIANS_DIR", "./debug_vis/canonical_gaussians")
        # os.makedirs(vis_dir, exist_ok=True)
        # for b_idx, gs_model in enumerate(gs_model_list):
        #     save_path = os.path.join(vis_dir, f"canonical_gaussians_batch{b_idx}.obj")
        #     visualize_gaussians_as_ellipsoids(
        #         gs_model,
        #         save_path=save_path,
        #         n_rings=8,
        #         n_sectors=8,
        #         scale_factor=3.0,  # Scale up ellipsoids for visibility
        #         opacity_threshold=0.0,
        #         max_gaussians=None,  # Set to e.g. 5000 for faster visualization
        #     )
        # breakpoint()
        out = self.forward_animate_gs(gs_model_list, query_points, flame_data, w2c, intrinsic, height, width, background_color, debug)
        
        return out


def test_head():
    import cv2
    
    human_model_path = "./model_zoo/human_parametric_models"
    device = "cuda"
    
    from accelerate.utils import set_seed
    set_seed(1234)

    from lam.datasets.video_head import VideoHeadDataset
    root_dir = "./train_data/vfhq_vhap/export"
    meta_path = "./train_data/vfhq_vhap/label/valid_id_list.json"
    # root_dir = "./train_data/nersemble/export"
    # meta_path = "./train_data/nersemble/label/valid_id_list1.json"
    dataset = VideoHeadDataset(root_dirs=root_dir, meta_path=meta_path, sample_side_views=7,
                    render_image_res_low=512, render_image_res_high=512,
                    render_region_size=(512, 512), source_image_res=512,
                    enlarge_ratio=[0.8, 1.2],
                    debug=False)

    data = dataset[0]
    
    def get_flame_params(data):
        flame_params = {}        
        flame_keys = ['root_pose', 'body_pose', 'jaw_pose', 'leye_pose', 'reye_pose', 'lhand_pose', 'rhand_pose', 'expr', 'trans', 'betas',\
                      'rotation', 'neck_pose', 'eyes_pose', 'translation']
        for k, v in data.items():
            if k in flame_keys:
                # print(k, v.shape)
                flame_params[k] = data[k]
        return flame_params
    
    flame_data = get_flame_params(data)

    flame_data_tmp = {}
    for k, v in flame_data.items():
        flame_data_tmp[k] = v.unsqueeze(0).to(device)
        print(k, v.shape)
    flame_data = flame_data_tmp
    
    c2ws = data["c2ws"].unsqueeze(0).to(device)
    intrs = data["intrs"].unsqueeze(0).to(device)
    render_images = data["render_image"].numpy()    
    render_h = data["render_full_resolutions"][0, 0]
    render_w= data["render_full_resolutions"][0, 1]
    render_bg_colors = data["render_bg_colors"].unsqueeze(0).to(device)
    print("c2ws", c2ws.shape, "intrs", intrs.shape, intrs)

    gs_render = GS3DRenderer(human_model_path=human_model_path, subdivide_num=2, smpl_type="flame", 
                             feat_dim=64, query_dim=64, use_rgb=True, sh_degree=3, mlp_network_config=None,
                             xyz_offset_max_step=0.0001, expr_param_dim=10, shape_param_dim=10, 
                             fix_opacity=True, fix_rotation=True, clip_scaling=0.001, add_teeth=False)
    gs_render.to(device)
    
    out = gs_render.forward(gs_hidden_features=torch.zeros((1, 2048, 64)).float().to(device),
                      query_points=None,
                      flame_data=flame_data,
                      c2w=c2ws,
                      intrinsic=intrs,
                      height=render_h,
                      width=render_w,
                      background_color=render_bg_colors,
                      debug=False)

    os.makedirs("./debug_vis/gs_render", exist_ok=True)
    for k, v in out.items():
        if k == "comp_rgb_bg":
            print("comp_rgb_bg", v)
            continue
        for b_idx in range(len(v)):
            if k == "3dgs":
                for v_idx in range(len(v[b_idx])):
                    v[b_idx][v_idx].save_ply(f"./debug_vis/gs_render/{b_idx}_{v_idx}.ply")
                continue
            for v_idx in range(v.shape[1]):
                save_path = os.path.join("./debug_vis/gs_render", f"{b_idx}_{v_idx}_{k}.jpg")
                if "normal" in k:
                    img = ((v[b_idx, v_idx].permute(1, 2, 0).detach().cpu().numpy() + 1.0) / 2. * 255).astype(np.uint8)
                else:
                    img = (v[b_idx, v_idx].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
                print(v[b_idx, v_idx].shape, img.shape, save_path)
                if "mask" in k:
                    render_img = render_images[v_idx].transpose(1, 2, 0) * 255
                    blend_img = (render_images[v_idx].transpose(1, 2, 0) * 255 * 0.5 + np.tile(img, (1, 1, 3)) * 0.5).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(save_path, np.hstack([np.tile(img, (1, 1, 3)), render_img.astype(np.uint8), blend_img])[:, :, (2, 1, 0)])
                else:
                    print(save_path, k)
                    cv2.imwrite(save_path, img)



if __name__ == "__main__":
    test_head()
