"""
Visualization script for precomputed FLAME latent tokens.
The tokens have shape (20k, 1024) corresponding to FLAME mesh vertices.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch
from omegaconf import OmegaConf
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from lam.models.rendering.gs_renderer import GS3DRenderer

def load_tokens(token_path):
    """Load tokens from .npz file."""
    data = np.load(token_path)
    tokens = data['tokens']  # Shape: (20000, 1024)
    print(f"Loaded tokens with shape: {tokens.shape}")
    return tokens


def load_flame_params(flame_param_path):
    """Load FLAME parameters from a .frame file."""
    params = torch.load(flame_param_path, map_location='cpu')
    print(f"Loaded FLAME parameters with keys: {list(params.keys())}")
    return params


def get_flame_vertices(flame_param_path, config_path=None, device='cpu'):
    """
    Get FLAME mesh vertices (20k vertices) using actual FLAME parameters.
    
    Args:
        flame_param_path: Path to .frame file containing FLAME parameters
        config_path: Path to model config file
        device: Device to use ('cpu' or 'cuda')
    """
    # Default config if none provided
    if config_path is None:
        config_path = project_root / "configs" / "training" / "precompute_lam_cafca.yaml"
    
    cfg = OmegaConf.load(config_path)
    
    # Map config to GS3DRenderer constructor arguments
    renderer_kwargs = {
        'human_model_path': cfg.model.human_model_path,
        'subdivide_num': cfg.model.flame_subdivide_num,
        'smpl_type': cfg.model.flame_type,
        'feat_dim': cfg.model.transformer_dim,
        'query_dim': cfg.model.gs_query_dim,
        'use_rgb': cfg.model.gs_use_rgb,
        'sh_degree': cfg.model.gs_sh,
        'mlp_network_config': cfg.model.gs_mlp_network_config,
        'xyz_offset_max_step': cfg.model.gs_xyz_offset_max_step,
        'clip_scaling': cfg.model.gs_clip_scaling,
        'shape_param_dim': cfg.model.shape_param_dim,
        'expr_param_dim': cfg.model.expr_param_dim,
        'fix_opacity': cfg.model.fix_opacity,
        'fix_rotation': cfg.model.fix_rotation,
        'skip_decoder': True,  # We don't need the decoder for just getting vertices
        'add_teeth': cfg.model.add_teeth,
        'teeth_bs_flag': cfg.model.teeth_bs_flag,
        'oral_mesh_flag': cfg.model.oral_mesh_flag,
        'num_gaussians_per_vertex': cfg.model.num_gaussians_per_vertex,
        'scale_sphere': cfg.model.get('scale_sphere', False),
        'decode_with_extra_info': cfg.model.get('decode_with_extra_info', None),
        'gradient_checkpointing': False,
        'use_mesh_shading': cfg.model.get('use_mesh_shading', False),
        'render_rgb': cfg.model.get('render_rgb', True),
    }
    
    # Initialize GS3DRenderer which contains the FLAME model
    renderer = GS3DRenderer(**renderer_kwargs)
    
    # Load actual FLAME parameters from file
    flame_params = load_flame_params(flame_param_path)
    
    # Prepare FLAME parameters for the model
    batch_size = 1
    flame_params_batch = {}
    
    # Add batch dimension if needed
    for key in ['betas', 'expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation']:
        if key in flame_params:
            param = flame_params[key]
            if not torch.is_tensor(param):
                param = torch.tensor(param)
            # Add batch dimension if not present
            if param.ndim == 1:
                param = param.unsqueeze(0)
            flame_params_batch[key] = param.to(device)
        else:
            print(f"Warning: Key '{key}' not found in FLAME parameters, using zeros")
            # Fallback to zeros if key is missing
            if key == 'betas':
                flame_params_batch[key] = torch.zeros(batch_size, cfg.model.shape_param_dim).to(device)
            elif key == 'expr':
                flame_params_batch[key] = torch.zeros(batch_size, cfg.model.expr_param_dim).to(device)
            elif key == 'eyes_pose':
                flame_params_batch[key] = torch.zeros(batch_size, 6).to(device)
            else:
                flame_params_batch[key] = torch.zeros(batch_size, 3).to(device)
    
    positions, _, normals, faces = renderer.get_query_points(flame_params_batch, device=torch.device(device))
    vertices = positions[0].cpu().numpy()  # Shape: (N, 3)
    faces_np = faces.cpu().numpy()  # Shape: (F, 3)
    normals_np = normals[0].cpu().numpy()  # Shape: (N, 3)
    
    print(f"FLAME vertices shape: {vertices.shape}")
    print(f"FLAME faces shape: {faces_np.shape}")
    
    return vertices, faces_np, normals_np


def dimensionality_reduction_2d(tokens):
    """Reduce tokens to 2D using PCA, t-SNE, and UMAP."""
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    
    results = {}
    
    # PCA (fastest)
    print("Computing PCA...")
    pca = PCA(n_components=2)
    tokens_pca = pca.fit_transform(tokens)
    results['pca'] = {
        'data': tokens_pca,
        'explained_variance': pca.explained_variance_ratio_.sum()
    }
    
    # t-SNE (slower but better for clusters)
    print("Computing t-SNE (this may take a while)...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    tokens_tsne = tsne.fit_transform(tokens)
    results['tsne'] = {'data': tokens_tsne}
    
    # UMAP (if available)
    try:
        import umap
        print("Computing UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=-1)
        tokens_umap = reducer.fit_transform(tokens)
        results['umap'] = {'data': tokens_umap}
    except ImportError:
        print("UMAP not available. Install with: pip install umap-learn")
        results['umap'] = None
    
    return results


def visualize_2d_projections(tokens, output_dir):
    """Create 2D projection visualizations."""
    print("\n=== Creating 2D Projections ===")
    results = dimensionality_reduction_2d(tokens)
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # PCA
    ax = axes[0]
    scatter = ax.scatter(results['pca']['data'][:, 0], 
                        results['pca']['data'][:, 1], 
                        c=range(len(tokens)), 
                        s=1, 
                        alpha=0.5, 
                        cmap='viridis')
    ax.set_title(f"PCA Projection\nExplained variance: {results['pca']['explained_variance']:.2%}")
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax, label='Vertex Index')
    
    # t-SNE
    ax = axes[1]
    scatter = ax.scatter(results['tsne']['data'][:, 0], 
                        results['tsne']['data'][:, 1], 
                        c=range(len(tokens)), 
                        s=1, 
                        alpha=0.5, 
                        cmap='viridis')
    ax.set_title('t-SNE Projection')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax, label='Vertex Index')
    
    # UMAP
    ax = axes[2]
    if results['umap'] is not None:
        scatter = ax.scatter(results['umap']['data'][:, 0], 
                            results['umap']['data'][:, 1], 
                            c=range(len(tokens)), 
                            s=1, 
                            alpha=0.5, 
                            cmap='viridis')
        ax.set_title('UMAP Projection')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        plt.colorbar(scatter, ax=ax, label='Vertex Index')
    else:
        ax.text(0.5, 0.5, 'UMAP not available', ha='center', va='center')
        ax.set_title('UMAP Projection (Not Available)')
    
    plt.tight_layout()
    save_path = output_dir / "2d_projections.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved 2D projections to: {save_path}")
    plt.close()
    
    return results


def visualize_feature_statistics(tokens, output_dir):
    """Visualize feature statistics."""
    print("\n=== Creating Feature Statistics ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Feature magnitude distribution
    feature_norms = np.linalg.norm(tokens, axis=1)
    axes[0, 0].hist(feature_norms, bins=100, alpha=0.7, edgecolor='black')
    axes[0, 0].set_title('Distribution of Feature Magnitudes')
    axes[0, 0].set_xlabel('L2 Norm')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Mean feature values across all points
    mean_features = tokens.mean(axis=0)
    axes[0, 1].plot(mean_features, linewidth=0.5)
    axes[0, 1].set_title('Mean Feature Values (per dimension)')
    axes[0, 1].set_xlabel('Feature Dimension')
    axes[0, 1].set_ylabel('Mean Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Feature variance
    feature_std = tokens.std(axis=0)
    axes[1, 0].plot(feature_std, linewidth=0.5)
    axes[1, 0].set_title('Feature Standard Deviation (per dimension)')
    axes[1, 0].set_xlabel('Feature Dimension')
    axes[1, 0].set_ylabel('Std Dev')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Feature value distribution (histogram for first feature dimension)
    axes[1, 1].hist(tokens[:, 0], bins=100, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Distribution of Feature Dimension 0')
    axes[1, 1].set_xlabel('Feature Value')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / "feature_statistics.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved feature statistics to: {save_path}")
    plt.close()


def visualize_3d_mesh_with_features(vertices, faces, tokens, output_dir):
    """Create 3D visualization of FLAME mesh colored by token features."""
    print("\n=== Creating 3D Mesh Visualizations ===")
    
    from sklearn.decomposition import PCA
    
    # Reduce features to 3D for RGB coloring
    pca_3d = PCA(n_components=3)
    colors_3d = pca_3d.fit_transform(tokens)
    
    # Normalize to [0, 1] for RGB
    colors_3d = (colors_3d - colors_3d.min(axis=0)) / (colors_3d.max(axis=0) - colors_3d.min(axis=0))
    
    # Create matplotlib 3D scatter plot
    fig = plt.figure(figsize=(15, 5))
    
    # View 1: Front view
    ax1 = fig.add_subplot(131, projection='3d')
    scatter = ax1.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
                         c=colors_3d, s=1, alpha=0.6)
    ax1.set_title('FLAME Mesh - Front View\n(Colored by PCA of features)')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.view_init(elev=0, azim=0)
    
    # View 2: Side view
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c=colors_3d, s=1, alpha=0.6)
    ax2.set_title('FLAME Mesh - Side View\n(Colored by PCA of features)')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.view_init(elev=0, azim=90)
    
    # View 3: Top view
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], 
               c=colors_3d, s=1, alpha=0.6)
    ax3.set_title('FLAME Mesh - Top View\n(Colored by PCA of features)')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.view_init(elev=90, azim=0)
    
    plt.tight_layout()
    save_path = output_dir / "3d_mesh_views.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved 3D mesh views to: {save_path}")
    plt.close()
    
    # Create interactive Plotly visualization
    try:
        import plotly.graph_objects as go
        
        print("Creating interactive 3D visualization...")
        fig = go.Figure(data=[
            go.Scatter3d(
                x=vertices[:, 0],
                y=vertices[:, 1],
                z=vertices[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=colors_3d,  # RGB from first 3 PCA components
                    opacity=0.8
                ),
                text=[f'Vertex {i}' for i in range(len(vertices))],
                hovertemplate='<b>%{text}</b><br>X: %{x:.3f}<br>Y: %{y:.3f}<br>Z: %{z:.3f}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title='Interactive 3D FLAME Mesh (Colored by Token Features)',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            ),
            width=1000,
            height=800
        )
        
        save_path_html = output_dir / "3d_mesh_interactive.html"
        fig.write_html(save_path_html)
        print(f"Saved interactive 3D visualization to: {save_path_html}")
        print(f"  -> Open this file in a web browser to interact with the 3D mesh!")
        
    except ImportError:
        print("Plotly not available. Install with: pip install plotly")


def visualize_per_dimension_spatial_distribution(vertices, tokens, output_dir, num_dims=6):
    """Visualize how specific feature dimensions vary spatially on the mesh."""
    print(f"\n=== Creating Spatial Distribution for {num_dims} Feature Dimensions ===")
    
    ncols = 3
    nrows = (num_dims + ncols - 1) // ncols
    fig = plt.figure(figsize=(15, 5 * nrows))
    
    for i in range(num_dims):
        ax = fig.add_subplot(nrows, ncols, i+1, projection='3d')
        
        feature_values = tokens[:, i]
        scatter = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2],
                           c=feature_values, s=1, alpha=0.6, cmap='coolwarm')
        
        ax.set_title(f'Feature Dimension {i}\nSpatial Distribution')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=10, azim=45)
        
        plt.colorbar(scatter, ax=ax, shrink=0.5, label=f'Dim {i} Value')
    
    plt.tight_layout()
    save_path = output_dir / "spatial_feature_distribution.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved spatial feature distribution to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize precomputed FLAME latent tokens')
    parser.add_argument('--token_path', type=str, required=True,
                       help='Path to the .npz file containing tokens')
    parser.add_argument('--flame_params', type=str, required=True,
                       help='Path to the .frame file containing FLAME parameters')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for visualizations (default: same as token_path)')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to config file (default: configs/training/precompute_lam_cafca.yaml)')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use for FLAME model (cpu or cuda)')
    parser.add_argument('--skip_2d', action='store_true',
                       help='Skip 2D projection visualizations (t-SNE can be slow)')
    
    args = parser.parse_args()
    
    # Setup paths
    token_path = Path(args.token_path)
    if not token_path.exists():
        print(f"Error: Token file not found: {token_path}")
        return
    
    flame_param_path = Path(args.flame_params)
    if not flame_param_path.exists():
        print(f"Error: FLAME parameter file not found: {flame_param_path}")
        return
    
    if args.output_dir is None:
        output_dir = token_path.parent / "visualizations"
    else:
        output_dir = Path(args.output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load tokens
    print(f"\nLoading tokens from: {token_path}")
    tokens = load_tokens(token_path)
    
    # Get FLAME vertices
    print("\nLoading FLAME mesh...")
    vertices, faces, normals = get_flame_vertices(flame_param_path, args.config, args.device)
    
    # Verify shapes match
    if len(vertices) != len(tokens):
        print(f"Warning: Number of vertices ({len(vertices)}) != number of tokens ({len(tokens)})")
        print(f"Using minimum of the two...")
        min_len = min(len(vertices), len(tokens))
        vertices = vertices[:min_len]
        tokens = tokens[:min_len]
    
    # Create visualizations
    print("\n" + "="*60)
    print("CREATING VISUALIZATIONS")
    print("="*60)
    
    # 1. Feature statistics
    visualize_feature_statistics(tokens, output_dir)
    
    # 2. 3D mesh visualizations
    visualize_3d_mesh_with_features(vertices, faces, tokens, output_dir)
    
    # 3. Spatial distribution of specific dimensions
    visualize_per_dimension_spatial_distribution(vertices, tokens, output_dir, num_dims=6)
    
    # 4. 2D projections (optional, can be slow)
    if not args.skip_2d:
        visualize_2d_projections(tokens, output_dir)
    else:
        print("\nSkipping 2D projections (use --skip_2d=False to enable)")
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE!")
    print("="*60)
    print(f"\nAll visualizations saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()

