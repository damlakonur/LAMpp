#!/usr/bin/env python3
"""
Interactive Gradio-based Gaussian Avatar Player
Works on remote machines - access via web browser!
"""

import gradio as gr
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf
import math
from PIL import Image

from lam.models.modeling_lam import ModelLAM
from lam.training.lightning_lam_cafca import LamLightningModel
from lam.dataset.cafca_lam_de_dataset_static import CafcaLamDataset


class GradioAvatarPlayer:
    def __init__(self, 
                 checkpoint_path,
                 config_path="configs/inference/inference_lam_cafca.yaml",
                 subject_id=1,
                 source_cam_ids=["C00", "C10"],
                 env_id="env_000",
                 expr_id="expr_00000",
                 render_size=512):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.render_size = render_size
        
        print(f"🚀 Initializing Avatar Player on {self.device}")
        
        # Load config
        self.config = OmegaConf.load(config_path)
        self.config.model["instantiate_encoder"] = False
        self.config.model["instantiate_transformer"] = False
        
        # Load dataset
        print(f"📁 Loading dataset for subject {subject_id}...")
        self.dataset = CafcaLamDataset(
            subject_list=[subject_id],
            num_source_frames=2,
            num_driving_frames=1,
            image_size=self.config.training.image_size,
            is_val=True,
            max_cache_size=16
        )
        
        # Load source data
        print(f"🎬 Loading source views: {source_cam_ids}...")
        expr_key = (subject_id, env_id, expr_id)
        expr_info = self.dataset.expression_data[expr_key]
        all_frames = expr_info['frames']
        cam_id_to_frame = {f["cam_id"]: f for f in all_frames}
        
        src_frames = [cam_id_to_frame[cam_id] for cam_id in source_cam_ids]
        
        # Load source frames EXACTLY like inference.py lines 308-309
        tokens = torch.stack([self.dataset._get_token_tensor(f["token_file_path"]) for f in src_frames])
        src_intrs_list = []
        src_w2cs_list = []
        render_intrs_list = []
        render_w2cs_list = []
        
        for f in src_frames:
            # Source intrinsics: 3x3
            src_intrs_list.append(torch.from_numpy(f["intrinsic_np"]).float())
            src_w2cs_list.append(torch.from_numpy(f["world_2_cam_np"]).float())
            
            # Render intrinsics: pad to 4x4 (line 320 in inference.py uses _stack_frame_list which pads)
            intr_np = f["intrinsic_np"]
            intr_4x4 = torch.eye(4, dtype=torch.float32)
            intr_4x4[:intr_np.shape[0], :intr_np.shape[1]] = torch.from_numpy(intr_np)
            render_intrs_list.append(intr_4x4)
            render_w2cs_list.append(torch.from_numpy(f["world_2_cam_np"]).float())
        
        src_intrs = torch.stack(src_intrs_list)
        src_w2cs = torch.stack(src_w2cs_list)
        render_intrs = torch.stack(render_intrs_list)
        render_w2cs = torch.stack(render_w2cs_list)
        
        # Load FLAME params
        flame_params = self.dataset._load_subject_flame_params(expr_info['flame_params_path'])
        betas = flame_params['betas'].clone()
        canon_2_cam = flame_params['canon_2_cam'].repeat(len(src_frames), 1, 1)
        
        # Store with batch dimension like inference.py line 346-356
        self.latent_points = tokens.unsqueeze(0).to(self.device)  # [1, N, 20018, 1024]
        self.src_intrs = src_intrs.unsqueeze(0).to(self.device)  # [1, N, 3, 3]
        self.src_w2cs = src_w2cs.unsqueeze(0).to(self.device)  # [1, N, 4, 4]
        self.src_canon_2_cam = canon_2_cam.unsqueeze(0).to(self.device)  # [1, N, 4, 4]
        self.shape_param = betas.to(self.device)  # [300]
        # Store all source flame params for use in inference
        self.src_flame_params = {k: v.to(self.device) for k, v in flame_params.items()}
        
        # Store ALL source views as potential render cameras (can switch between them)
        self.render_w2cs = render_w2cs.unsqueeze(0).to(self.device)  # [1, N, 4, 4]
        self.render_intrs = render_intrs.unsqueeze(0).to(self.device)  # [1, N, 4, 4]
        self.default_render_idx = 0  # Use first source view by default
        
        print(f"✓ Loaded tokens: {self.latent_points.shape}")
        
        # Load model
        print(f"🤖 Loading model from {checkpoint_path}...")
        from lam.training.inference import load_checkpoint_with_shape_mismatch_handling
        # self.lit = load_checkpoint_with_shape_mismatch_handling(checkpoint_path, self.config)
        self.lit = LamLightningModel.load_from_checkpoint(checkpoint_path)
        self.lit = self.lit.to(self.device).eval()
        
        for p in self.lit.parameters():
            p.requires_grad_(False)
        
        print("✅ Setup complete!")
    
    def apply_orbit_transform(self, base_w2c, azimuth, elevation, distance_scale):
        """Apply orbit transformation to existing camera matrix using rotation matrices
        
        Args:
            base_w2c: Base world-to-camera matrix [4, 4]
            azimuth: Rotation around Y-axis in degrees (-180 to 180)
            elevation: Rotation around X-axis in degrees (-45 to 45)
            distance_scale: Scale factor for camera distance (0.5-2.5)
        """
        base_w2c_np = base_w2c.cpu().numpy().astype(np.float32)
        
        # Convert to c2w for easier manipulation
        base_c2w = np.linalg.inv(base_w2c_np)
        
        # Extract base camera position and rotation
        base_cam_pos = base_c2w[:3, 3].copy()
        base_distance = np.linalg.norm(base_cam_pos)
        
        # Create rotation matrices for azimuth (around Y) and elevation (around X)
        az_rad = np.radians(azimuth)
        el_rad = np.radians(elevation)
        
        # Rotation around Y-axis (azimuth)
        R_y = np.array([
            [np.cos(az_rad), 0, np.sin(az_rad), 0],
            [0, 1, 0, 0],
            [-np.sin(az_rad), 0, np.cos(az_rad), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Rotation around X-axis (elevation)
        R_x = np.array([
            [1, 0, 0, 0],
            [0, np.cos(el_rad), -np.sin(el_rad), 0],
            [0, np.sin(el_rad), np.cos(el_rad), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Apply distance scaling
        scale = distance_scale
        S = np.array([
            [scale, 0, 0, 0],
            [0, scale, 0, 0],
            [0, 0, scale, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Combine transformations: First scale, then rotate around X (elevation), then rotate around Y (azimuth)
        # Apply to base c2w
        new_c2w = R_y @ R_x @ S @ base_c2w
        
        # Convert back to w2c
        new_w2c = np.linalg.inv(new_c2w)
        
        return torch.from_numpy(new_w2c).float()
    
    def render(self, 
               expr_0, expr_1, expr_2, expr_3, expr_4,
               expr_5, expr_6, expr_7, expr_8, expr_9,
               jaw_x, jaw_y, jaw_z,
               neck_x, neck_y, neck_z,
               azimuth, elevation, distance):
        """Render avatar with given parameters"""
        
        with torch.no_grad():
            # Build FLAME params
            expr = torch.zeros(1, 100).to(self.device)
            expr[0, 0] = expr_0
            expr[0, 1] = expr_1
            expr[0, 2] = expr_2
            expr[0, 3] = expr_3
            expr[0, 4] = expr_4
            expr[0, 5] = expr_5
            expr[0, 6] = expr_6
            expr[0, 7] = expr_7
            expr[0, 8] = expr_8
            expr[0, 9] = expr_9
            
            # Build flame_params
            # Structure from gs_renderer.py get_sing_batch_smpl_data():
            # - betas: [batch, D] - batch only
            # - All others: [batch, num_views, D] - batch AND num_views (will be indexed to drop batch)
            flame_params = {
                'betas': self.src_flame_params['betas'].unsqueeze(0) if self.src_flame_params['betas'].ndim == 1 else self.src_flame_params['betas'],  # [1, 300]
                'expr': expr.unsqueeze(1),  # [1, 100] -> [1, 1, 100]
                'rotation': self.src_flame_params['rotation'].unsqueeze(0).unsqueeze(0) if self.src_flame_params['rotation'].ndim == 1 else self.src_flame_params['rotation'].unsqueeze(0),  # [1, 1, 3]
                'neck_pose': torch.tensor([[[neck_x, neck_y, neck_z]]]).to(self.device),  # [1, 1, 3]
                'jaw_pose': torch.tensor([[[jaw_x, jaw_y, jaw_z]]]).to(self.device),  # [1, 1, 3]
                'eyes_pose': self.src_flame_params['eyes_pose'].unsqueeze(0).unsqueeze(0) if self.src_flame_params['eyes_pose'].ndim == 1 else self.src_flame_params['eyes_pose'].unsqueeze(0),  # [1, 1, 6]
                'translation': self.src_flame_params['translation'].unsqueeze(0).unsqueeze(0) if self.src_flame_params['translation'].ndim == 1 else self.src_flame_params['translation'].unsqueeze(0),  # [1, 1, 3]
            }
            
            # Apply orbit transformation to base camera
            base_w2c = self.render_w2cs[0, self.default_render_idx]  # [4, 4]
            custom_w2c = self.apply_orbit_transform(base_w2c, azimuth, elevation, distance)
            render_w2cs = custom_w2c.unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, 4, 4]
            
            # Use intrinsics from first source camera
            render_intrs = self.render_intrs[:, self.default_render_idx:self.default_render_idx+1]  # [1, 1, 4, 4]
            bg_color = torch.ones((1, 1, 3), device=self.device)
            
            try:
                # Call model EXACTLY like inference.py line 359
                with torch.no_grad():
                    batch = {
                        "src_w2cs": self.src_w2cs,
                        "src_intrs": self.src_intrs,
                        "source_canon_2_cam": self.src_canon_2_cam,
                        "render_w2cs": render_w2cs,
                        "render_intrs": render_intrs,
                        "render_bg_colors": bg_color,
                        "flame_params": flame_params,
                        "latent_points": self.latent_points,
                    }
                    output = self.lit(batch)
                
                # Extract RGB: output is [B, Nv, C, H, W] (see lightning_lam_cafca.py line 224)
                rgb = output['comp_rgb'][0, 0]  # [C, H, W] = [3, 512, 512]
                rgb = rgb.permute(1, 2, 0)  # [C, H, W] -> [H, W, C]
                rgb = rgb.cpu().numpy()
                rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)
                return Image.fromarray(rgb)
                
            except Exception as e:
                print(f"❌ Render error: {e}")
                import traceback
                traceback.print_exc()
                # Return error image with text
                error_img = np.ones((self.render_size, self.render_size, 3), dtype=np.uint8) * 200
                return Image.fromarray(error_img)


def create_interface(player):
    """Create Gradio interface"""
    
    with gr.Blocks(title="Gaussian Avatar Player") as demo:
        gr.Markdown("# 🎭 Interactive Gaussian Avatar Player")
        gr.Markdown("Adjust FLAME expression parameters and camera to animate your avatar in real-time!")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📷 Camera (Free View)")
                azimuth = gr.Slider(-180, 180, value=0, step=5, label="Azimuth (horizontal rotation)")
                elevation = gr.Slider(-45, 45, value=0, step=5, label="Elevation (vertical angle)")
                distance = gr.Slider(0.5, 2.5, value=1.0, step=0.1, label="Distance (1.0 = default)")
                gr.Markdown("### 😊 Expression Parameters (0-9)")
                expr_sliders = []
                for i in range(10):
                    slider = gr.Slider(-3, 3, value=0, step=0.1, label=f"expr_{i}")
                    expr_sliders.append(slider)
                
                # gr.Markdown("### 🦴 Pose Parameters")
                # with gr.Row():
                #     jaw_x = gr.Slider(-1, 1, value=0, step=0.05, label="Jaw X")
                #     jaw_y = gr.Slider(-1, 1, value=0, step=0.05, label="Jaw Y")
                #     jaw_z = gr.Slider(-1, 1, value=0, step=0.05, label="Jaw Z")
                
                # with gr.Row():
                #     neck_x = gr.Slider(-1, 1, value=0, step=0.05, label="Neck X")
                #     neck_y = gr.Slider(-1, 1, value=0, step=0.05, label="Neck Y")
                #     neck_z = gr.Slider(-1, 1, value=0, step=0.05, label="Neck Z")
                
                
                reset_btn = gr.Button("🔄 Reset All", variant="secondary")
            
            with gr.Column(scale=1):
                output_image = gr.Image(label="Rendered Avatar", type="pil")
                gr.Markdown("**🔴 Live Preview**: Rendering updates automatically when you adjust any slider")
                gr.Markdown("### 🦴 Pose Parameters")
                with gr.Row():
                    jaw_x = gr.Slider(-1, 1, value=0, step=0.05, label="Jaw X")
                    jaw_y = gr.Slider(-1, 1, value=0, step=0.05, label="Jaw Y")
                    jaw_z = gr.Slider(-1, 1, value=0, step=0.05, label="Jaw Z")
                
                with gr.Row():
                    neck_x = gr.Slider(-1, 1, value=0, step=0.05, label="Neck X")
                    neck_y = gr.Slider(-1, 1, value=0, step=0.05, label="Neck Y")
                    neck_z = gr.Slider(-1, 1, value=0, step=0.05, label="Neck Z")
        
        # Automatic rendering on any slider change
        all_inputs = expr_sliders + [jaw_x, jaw_y, jaw_z, neck_x, neck_y, neck_z, azimuth, elevation, distance]
        
        # Connect all sliders to trigger automatic rendering
        for slider in all_inputs:
            slider.change(
                fn=player.render,
                inputs=all_inputs,
                outputs=output_image
            )
        
        # Reset button
        def reset():
            return [0] * 10 + [0, 0, 0, 0, 0, 0, 0, 0, 1.0]
        
        reset_btn.click(
            fn=reset,
            inputs=[],
            outputs=all_inputs
        )
        
        # Initial render on load
        def initial_render():
            print("🎨 Performing initial render...")
            result = player.render(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0)
            print(f"✅ Initial render complete: {type(result)}")
            return result
        
        demo.load(
            fn=initial_render,
            inputs=None,
            outputs=output_image
        )
    
    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", type=str, required=True)
    parser.add_argument("--config-path", type=str, default="configs/inference/inference_lam_cafca.yaml")
    parser.add_argument("--subject-id", type=int, default=1)
    parser.add_argument("--source-cam-ids", nargs=2, default=["C00", "C10"])
    parser.add_argument("--env-id", type=str, default="env_000")
    parser.add_argument("--expr-id", type=str, default="expr_00000")
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="Create public link")
    
    args = parser.parse_args()
    
    # Initialize player
    player = GradioAvatarPlayer(
        checkpoint_path=args.checkpoint_path,
        config_path=args.config_path,
        subject_id=args.subject_id,
        source_cam_ids=args.source_cam_ids,
        env_id=args.env_id,
        expr_id=args.expr_id,
        render_size=args.render_size
    )
    
    # Create and launch interface
    demo = create_interface(player)
    
    print(f"\n🌐 Launching web interface on port {args.port}")
    print(f"📱 Access locally: http://localhost:{args.port}")
    if args.share:
        print(f"🌍 Public link will be generated...")
    
    demo.launch(
        server_name="0.0.0.0",  # Listen on all interfaces
        server_port=args.port,
        share=args.share  # Set True to get public link
    )

