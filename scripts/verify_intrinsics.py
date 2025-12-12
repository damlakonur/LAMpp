#!/usr/bin/env python3
"""
Verify intrinsics update by projecting FLAME mesh onto images.
This will render the FLAME mesh on both original (cropped) and processed (enlarged) images
to verify that the intrinsic updates are correct.

Usage:
    python verify_intrinsics.py --subject 13 --env env_000 --expr expr_00000 --cam C00
"""

import argparse
import json
import numpy as np
import torch
import cv2
from pathlib import Path
import sys

# Add LAMpp to path
sys.path.insert(0, str(Path(__file__).parent))

from dreifus.matrix import Intrinsics, Pose
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.pyvista import render_from_camera
from dreifus.render import project, draw_onto_image
import pyvista as pv


def load_flame_mesh(flame_params_path, device='cuda'):
    """Load FLAME mesh from .frame file."""
    from lam.models.rendering.flame_model.flame import FlameHead
    
    # Load params first to get correct dimensions
    params = torch.load(flame_params_path, map_location=device)
    
    # Get shape and expression param dimensions from loaded params
    n_shape = params['betas'].shape[0]
    n_expr = params['expr'].shape[0]
    
    print(f"  Loaded params: shape={n_shape}, expr={n_expr}")
    
    # Initialize FLAME model with correct dimensions
    human_model_path = './model_zoo/human_parametric_models'
    flame_model = FlameHead(
        shape_params=n_shape,
        expr_params=n_expr,
        flame_model_path=f'{human_model_path}/flame_assets/flame/flame2020.pkl',
        flame_lmk_embedding_path=f"{human_model_path}/flame_assets/flame/landmark_embedding_with_eyes.npy",
        flame_template_mesh_path=f"{human_model_path}/flame_assets/flame/head_template_mesh.obj",
        flame_parts_path=f"{human_model_path}/flame_assets/flame/FLAME_masks.pkl",
        add_teeth=False,
        add_shoulder=False,
    ).to(device)
    
    # Forward pass to get vertices  
    # Note: We need to manually apply transformations since lbs() has a signature issue
    # Instead, let's just use the template mesh with shape and expression
    with torch.no_grad():
        # Get shaped vertices with shape and expression params
        betas = torch.cat([params['betas'][:n_shape], params['expr'][:n_expr]], dim=0).unsqueeze(0)
        
        template_vertices = flame_model.v_template.unsqueeze(0)
        from lam.models.rendering.flame_model.lbs import blend_shapes
        v_shaped = template_vertices + blend_shapes(betas, flame_model.shapedirs)
        
        # For verification, we just need the shaped mesh in canonical space
        # The actual pose will be applied via canon_2_cam
        vertices = v_shaped[0].cpu().numpy()  # [N, 3]
    
    faces = flame_model.faces.cpu().numpy()  # [F, 3]
    
    # Get canon_2_cam if it exists
    canon_2_cam = params.get('canon_2_cam', torch.eye(4)).cpu().numpy()
    
    return vertices, faces, canon_2_cam


def project_vertices_to_image(vertices, canon_2_cam, cam_2_world, K, rgb_image):
    """
    Project FLAME vertices as points onto RGB image using dreifus.
    
    Args:
        vertices: [N, 3] canonical vertices
        canon_2_cam: [4, 4] canonical to camera space transform
        cam_2_world: [4, 4] camera to world transform  
        K: [3, 3] intrinsic matrix
        rgb_image: [H, W, 3] RGB image to overlay on
    
    Returns:
        RGB image with projected vertices as red dots
    """
    # Apply canon_2_cam to get vertices in camera space
    vertices_hom = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    vertices_cam = (canon_2_cam @ vertices_hom.T).T[:, :3]
    
    # Setup camera intrinsics and pose
    intr = Intrinsics(K)
    pose = Pose(
        cam_2_world,
        pose_type=PoseType.CAM_2_WORLD,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )
    
    # Project 3D vertices to image space
    proj_points = project(vertices_cam, pose, intr)  # (N, 3)
    
    # Draw projected points on image
    image_drawn = rgb_image.copy()
    draw_onto_image(image_drawn, proj_points, (255, 0, 0))  # red dots
    
    return image_drawn


def project_mesh_to_image(vertices, faces, canon_2_cam, cam_2_world, K, rgb_image, alpha=0.4):
    """
    Render FLAME mesh and overlay on RGB image using PyVista.
    
    Args:
        vertices: [N, 3] canonical vertices
        faces: [F, 3] faces
        canon_2_cam: [4, 4] canonical to camera space transform
        cam_2_world: [4, 4] camera to world transform  
        K: [3, 3] intrinsic matrix
        rgb_image: [H, W, 3] RGB image to overlay on
        alpha: transparency for GT image
    
    Returns:
        RGB image with mesh overlay
    """
    H, W = rgb_image.shape[:2]
    
    # Apply canon_2_cam to get vertices in camera space
    vertices_hom = np.concatenate([vertices, np.ones((vertices.shape[0], 1))], axis=1)
    vertices_cam = (canon_2_cam @ vertices_hom.T).T[:, :3]
    
    # Create PyVista mesh
    faces_pv = np.hstack([np.full((faces.shape[0], 1), 3), faces])
    mesh = pv.PolyData(vertices_cam, faces_pv)
    
    # Setup renderer
    p = pv.Plotter(off_screen=True, window_size=(W, H))
    p.background_color = "white"
    p.add_mesh(mesh, color="green", opacity=1.0)
    
    # Setup camera intrinsics
    intr = Intrinsics(K)
    
    # Setup camera pose (cam_2_world)
    cam_2_world_pose = Pose(
        cam_2_world,
        pose_type=PoseType.CAM_2_WORLD,
        camera_coordinate_convention=CameraCoordinateConvention.OPEN_CV,
    )
    
    # Render
    img_render = render_from_camera(p, cam_2_world_pose, intr)  # RGBA
    p.close()
    
    # Convert render to RGB
    if img_render.shape[2] == 4:
        img_render = cv2.cvtColor(img_render, cv2.COLOR_RGBA2RGB)
    
    # Resize if needed
    if img_render.shape[:2] != (H, W):
        img_render = cv2.resize(img_render, (W, H))
    
    # Overlay with GT image
    beta = 1.0 - alpha
    overlay = cv2.addWeighted(rgb_image, alpha, img_render, beta, 0)
    
    return overlay


def verify_subject_cam(root_dir, subject_id, env_id, expr_id, cam_id):
    """
    Verify intrinsics for one camera view.
    """
    subject_str = str(subject_id).zfill(5)
    base_path = Path(root_dir) / subject_str / env_id / expr_id
    
    # Paths
    flame_path = base_path / "00400.frame"
    original_cam_path = base_path / "cameras_json" / f"{cam_id}.json"
    processed_cam_path = base_path / "processed_cameras_json" / f"{cam_id}.json"
    original_img_path = base_path / "frames" / "cropped_images" / f"{cam_id}.jpg"
    processed_img_path = base_path / "frames" / "processed_images" / f"{cam_id}.jpg"
    
    print(f"Verifying: Subject {subject_id}, {env_id}, {expr_id}, Camera {cam_id}")
    print("="*80)
    
    # Check files exist
    for path, name in [(flame_path, "FLAME params"),
                       (original_cam_path, "Original camera"),
                       (processed_cam_path, "Processed camera"),
                       (original_img_path, "Original image"),
                       (processed_img_path, "Processed image")]:
        if not path.exists():
            print(f"❌ {name} not found: {path}")
            return
        print(f"✓ {name}: {path.name}")
    
    print()
    
    # Load FLAME mesh
    print("Loading FLAME mesh...")
    vertices, faces, canon_2_cam = load_flame_mesh(flame_path)
    print(f"  Vertices: {vertices.shape}")
    print(f"  Faces: {faces.shape}")
    print(f"  Canon2Cam shape: {canon_2_cam.shape}")
    print()
    
    # Load camera params
    with open(original_cam_path) as f:
        original_cam = json.load(f)
    with open(processed_cam_path) as f:
        processed_cam = json.load(f)
    
    K_original = np.array(original_cam['K'])
    K_processed = np.array(processed_cam['K'])
    cam_2_world = np.array(original_cam['cam2world'])
    
    print("Original Intrinsics:")
    print(K_original)
    print(f"  fx={K_original[0,0]:.2f}, fy={K_original[1,1]:.2f}")
    print(f"  cx={K_original[0,2]:.2f}, cy={K_original[1,2]:.2f}")
    print()
    
    print("Processed Intrinsics:")
    print(K_processed)
    print(f"  fx={K_processed[0,0]:.2f}, fy={K_processed[1,1]:.2f}")
    print(f"  cx={K_processed[0,2]:.2f}, cy={K_processed[1,2]:.2f}")
    print()
    
    if 'crop_offset_x' in processed_cam:
        print(f"Crop info:")
        print(f"  Offset: ({processed_cam['crop_offset_x']}, {processed_cam['crop_offset_y']})")
        print(f"  Crop size: {processed_cam['crop_width']}x{processed_cam['crop_height']}")
        print()
    
    # Load images
    img_original = cv2.imread(str(original_img_path))
    img_processed = cv2.imread(str(processed_img_path))
    
    img_original_rgb = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)
    img_processed_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
    
    print(f"Original image shape: {img_original.shape}")
    print(f"Processed image shape: {img_processed.shape}")
    print()
    
    # METHOD 1: Render mesh overlay
    print("Rendering FLAME mesh on original image...")
    try:
        overlay_original_mesh = project_mesh_to_image(
            vertices, faces, canon_2_cam, cam_2_world, K_original, 
            img_original_rgb
        )
    except Exception as e:
        print(f"  Error rendering original: {e}")
        import traceback
        traceback.print_exc()
        overlay_original_mesh = img_original_rgb
    
    print("Rendering FLAME mesh on processed image...")
    try:
        overlay_processed_mesh = project_mesh_to_image(
            vertices, faces, canon_2_cam, cam_2_world, K_processed,
            img_processed_rgb
        )
    except Exception as e:
        print(f"  Error rendering processed: {e}")
        import traceback
        traceback.print_exc()
        overlay_processed_mesh = img_processed_rgb
    
    # METHOD 2: Project vertices as points
    print("Projecting vertices on original image...")
    try:
        overlay_original_points = project_vertices_to_image(
            vertices, canon_2_cam, cam_2_world, K_original,
            img_original_rgb
        )
    except Exception as e:
        print(f"  Error projecting original: {e}")
        import traceback
        traceback.print_exc()
        overlay_original_points = img_original_rgb
    
    print("Projecting vertices on processed image...")
    try:
        overlay_processed_points = project_vertices_to_image(
            vertices, canon_2_cam, cam_2_world, K_processed,
            img_processed_rgb
        )
    except Exception as e:
        print(f"  Error projecting processed: {e}")
        import traceback
        traceback.print_exc()
        overlay_processed_points = img_processed_rgb
    
    # Create comparison images
    comparison_mesh = np.hstack([overlay_original_mesh, overlay_processed_mesh])
    comparison_points = np.hstack([overlay_original_points, overlay_processed_points])
    
    # Add labels to mesh comparison
    cv2.putText(comparison_mesh, "Original (Cropped) - Mesh Render", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(comparison_mesh, "Processed (Enlarged) - Mesh Render", (img_original.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Add labels to points comparison
    cv2.putText(comparison_points, "Original (Cropped) - Point Projection", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    cv2.putText(comparison_points, "Processed (Enlarged) - Point Projection", (img_original.shape[1] + 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save both comparison images
    output_path_mesh = Path(f"verify_intrinsics_mesh_{subject_id}_{env_id}_{expr_id}_{cam_id}.jpg")
    output_path_points = Path(f"verify_intrinsics_points_{subject_id}_{env_id}_{expr_id}_{cam_id}.jpg")
    
    comparison_mesh_bgr = cv2.cvtColor(comparison_mesh, cv2.COLOR_RGB2BGR)
    comparison_points_bgr = cv2.cvtColor(comparison_points, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(str(output_path_mesh), comparison_mesh_bgr)
    cv2.imwrite(str(output_path_points), comparison_points_bgr)
    
    print(f"\n✓ Verification images saved:")
    print(f"  Mesh render: {output_path_mesh}")
    print(f"  Point projection: {output_path_points}")
    print(f"\n{'='*80}")
    print(f"If the FLAME mesh/points align correctly in BOTH images, intrinsics are correct!")
    print(f"Check the saved images above.")
    print(f"{'='*80}")


def main():
    parser = argparse.ArgumentParser(description="Verify intrinsics by projecting FLAME mesh")
    parser.add_argument('--subject', type=int, required=True, help='Subject ID')
    parser.add_argument('--env', type=str, default='env_000', help='Environment ID')
    parser.add_argument('--expr', type=str, default='expr_00000', help='Expression ID')
    parser.add_argument('--cam', type=str, default='C00', help='Camera ID')
    parser.add_argument('--root-dir', type=str, default='/home/cafca_dataset',
                       help='Root directory of CAFCA dataset')
    
    args = parser.parse_args()
    
    verify_subject_cam(
        args.root_dir,
        args.subject,
        args.env,
        args.expr,
        args.cam
    )


if __name__ == "__main__":
    main()

