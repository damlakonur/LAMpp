#!/usr/bin/env python3
"""
CAFCA Dataset Image Enlargement and Intrinsics Update Script

Usage:
    python process_cafca_enlargement.py --subjects 1 30 45
    python process_cafca_enlargement.py --subjects 1  # single subject
    python process_cafca_enlargement.py --all  # process all subjects
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
from typing import List, Tuple, Optional


class CafcaEnlargementProcessor:
    def __init__(self, 
                 root_dir: str = "/home/cafca_dataset",
                 enlarge_ratio: float = 1.15,
                 output_size: int = 512):
        """
        Args:
            root_dir: Root directory of CAFCA dataset
            enlarge_ratio: How much to enlarge crop (1.15 = 15% padding)
            output_size: Final image size (512x512)
        """
        self.root_dir = Path(root_dir)
        self.enlarge_ratio = enlarge_ratio
        self.output_size = output_size
    
    def center_crop_according_to_mask(self, 
                                      img: np.ndarray, 
                                      mask: np.ndarray) -> Tuple[np.ndarray, int, int]:
        """
        Crop image centered on mask bounding box.
        
        Args:
            img: [H, W, 3] RGB image
            mask: [H, W] binary mask (255=head, 0=background)
            
        Returns:
            cropped_img, offset_x, offset_y
        """
        # Find mask bounding box
        ys, xs = np.where(mask > 128)
        
        if len(xs) == 0 or len(ys) == 0:
            raise ValueError("Empty mask - no head pixels found")
        
        x_min, x_max = np.min(xs), np.max(xs)
        y_min, y_max = np.min(ys), np.max(ys)
        
        # Calculate center
        center_x = (x_min + x_max) / 2
        center_y = (y_min + y_max) / 2
        
        # Make square crop (max dimension + enlargement)
        width = x_max - x_min
        height = y_max - y_min
        max_dim = max(width, height)
        half_size = (max_dim * self.enlarge_ratio) / 2
        
        # Calculate crop bounds
        crop_x_start = int(center_x - half_size)
        crop_y_start = int(center_y - half_size)
        crop_x_end = int(center_x + half_size)
        crop_y_end = int(center_y + half_size)
        
        # Clamp to image bounds
        H, W = img.shape[:2]
        crop_x_start = max(0, crop_x_start)
        crop_y_start = max(0, crop_y_start)
        crop_x_end = min(W, crop_x_end)
        crop_y_end = min(H, crop_y_end)
        
        # Crop
        cropped_img = img[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
        
        return cropped_img, crop_x_start, crop_y_start
    
    def update_intrinsics(self, 
                          K: np.ndarray,
                          offset_x: int,
                          offset_y: int,
                          crop_w: int,
                          crop_h: int) -> np.ndarray:
        """
        Update intrinsics after crop and resize.
        
        Args:
            K: 3x3 intrinsic matrix
            offset_x, offset_y: crop offset
            crop_w, crop_h: crop dimensions before resize
            
        Returns:
            Updated 3x3 intrinsic matrix
        """
        K_new = K.copy()
        
        # Step 1: Subtract crop offset (principal point shifts)
        K_new[0, 2] -= offset_x  # cx' = cx - left
        K_new[1, 2] -= offset_y  # cy' = cy - top
        
        # Step 2: Scale for resize to output_size x output_size
        scale_x = self.output_size / crop_w
        scale_y = self.output_size / crop_h
        K_new[0, 0] *= scale_x  # fx'
        K_new[1, 1] *= scale_y  # fy'
        K_new[0, 2] *= scale_x  # cx''
        K_new[1, 2] *= scale_y  # cy''
        
        # Note: We do NOT override cx, cy to be at the center because our crop
        # is centered on the mask bounding box, not the image center.
        # The scaled principal point should be correct as-is.
        
        return K_new
    
    def process_single_frame(self,
                            image_path: Path,
                            mask_path: Path,
                            camera_path: Path,
                            output_img_path: Path,
                            output_cam_path: Path,
                            output_mask_path: Path) -> bool:
        """
        Process a single frame: crop, resize, update intrinsics.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load image and mask
            img = cv2.imread(str(image_path))
            if img is None:
                print(f"  Warning: Failed to load image {image_path}")
                return False
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print(f"  Warning: Failed to load mask {mask_path}")
                return False
            
            # Load camera params
            with open(camera_path, 'r') as f:
                cam_params = json.load(f)
            
            K_original = np.array(cam_params["K"])
            
            # Crop image centered on head mask
            cropped_img, offset_x, offset_y = self.center_crop_according_to_mask(img, mask)
            crop_h, crop_w = cropped_img.shape[:2]
            
            # Apply same crop to mask
            crop_x_end = offset_x + crop_w
            crop_y_end = offset_y + crop_h
            cropped_mask = mask[offset_y:crop_y_end, offset_x:crop_x_end]
            
            # Resize to output_size x output_size
            resized_img = cv2.resize(cropped_img, 
                                    (self.output_size, self.output_size), 
                                    interpolation=cv2.INTER_AREA)
            
            resized_mask = cv2.resize(cropped_mask,
                                     (self.output_size, self.output_size),
                                     interpolation=cv2.INTER_NEAREST)  # Use NEAREST for binary mask
            
            # Update intrinsics
            K_new = self.update_intrinsics(K_original, offset_x, offset_y, crop_w, crop_h)
            
            # Save processed image as JPG
            output_img_path.parent.mkdir(parents=True, exist_ok=True)
            resized_img_bgr = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
            # Use high quality JPG (95)
            cv2.imwrite(str(output_img_path), resized_img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            # Save processed mask as PNG
            output_mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_mask_path), resized_mask)
            
            # Update and save camera params
            cam_params_new = cam_params.copy()
            cam_params_new["K"] = K_new.tolist()
            cam_params_new["image_size_x"] = self.output_size
            cam_params_new["image_size_y"] = self.output_size
            cam_params_new["focal_length"] = float(K_new[0, 0])
            cam_params_new["principal_point_x"] = float(K_new[0, 2])
            cam_params_new["principal_point_y"] = float(K_new[1, 2])
            cam_params_new["original_K"] = K_original.tolist()
            cam_params_new["crop_offset_x"] = int(offset_x)
            cam_params_new["crop_offset_y"] = int(offset_y)
            cam_params_new["crop_width"] = int(crop_w)
            cam_params_new["crop_height"] = int(crop_h)
            
            output_cam_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_cam_path, 'w') as f:
                json.dump(cam_params_new, f, indent=2)
            
            return True
            
        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")
            return False
    
    def process_subject(self, subject_id: int) -> dict:
        """
        Process all frames for a subject.
        
        Returns:
            Dict with statistics
        """
        subject_str = str(subject_id).zfill(5)
        subject_dir = self.root_dir / subject_str
        
        if not subject_dir.exists():
            print(f"Subject directory not found: {subject_dir}")
            return {"total": 0, "success": 0, "failed": 0}
        
        stats = {"total": 0, "success": 0, "failed": 0}
        
        # Find all env_XXX directories
        env_dirs = sorted([d for d in subject_dir.iterdir() 
                          if d.is_dir() and d.name.startswith("env_")])
        
        for env_dir in env_dirs:
            # Find all expr_XXXXX directories
            expr_dirs = sorted([d for d in env_dir.iterdir() 
                               if d.is_dir() and d.name.startswith("expr_")])
            
            for expr_dir in expr_dirs:
                frames_dir = expr_dir / "frames"
                
                # Required directories
                cropped_images_dir = frames_dir / "cropped_images"
                head_masks_dir = frames_dir / "head_masks"
                cameras_dir = expr_dir / "cameras_json"  # cameras_json is at expr level, not frames level
                
                # Check if all required dirs exist
                if not all([cropped_images_dir.exists(), 
                           head_masks_dir.exists(), 
                           cameras_dir.exists()]):
                    continue
                
                # Output directories
                output_images_dir = frames_dir / "processed_images"  # Inside frames/
                output_masks_dir = frames_dir / "processed_masks"  # Inside frames/
                output_cameras_dir = expr_dir / "processed_cameras_json"  # At expr level (same as cameras_json)
                
                # Get list of images (both PNG and JPG)
                image_files = sorted(list(cropped_images_dir.glob("*.png")) + 
                                   list(cropped_images_dir.glob("*.jpg")))
                
                for img_file in image_files:
                    cam_id = img_file.stem
                    # Try both PNG and JPG for mask
                    mask_file = head_masks_dir / f"{cam_id}.png"
                    if not mask_file.exists():
                        mask_file = head_masks_dir / f"{cam_id}.jpg"
                    camera_file = cameras_dir / f"{cam_id}.json"
                    
                    output_img_file = output_images_dir / f"{cam_id}.jpg"  # Save as JPG
                    output_mask_file = output_masks_dir / f"{cam_id}.png"  # Save as PNG
                    output_cam_file = output_cameras_dir / f"{cam_id}.json"
                    
                    # Check if all files exist
                    if not all([img_file.exists(), mask_file.exists(), camera_file.exists()]):
                        continue
                    
                    stats["total"] += 1
                    
                    # Process frame
                    success = self.process_single_frame(
                        img_file, mask_file, camera_file,
                        output_img_file, output_cam_file, output_mask_file
                    )
                    
                    if success:
                        stats["success"] += 1
                    else:
                        stats["failed"] += 1
        
        return stats
    
    def process_subjects(self, subject_ids: List[int]):
        """
        Process multiple subjects.
        """
        print(f"Processing {len(subject_ids)} subjects with:")
        print(f"  - Enlarge ratio: {self.enlarge_ratio}")
        print(f"  - Output size: {self.output_size}x{self.output_size}")
        print()
        
        total_stats = {"total": 0, "success": 0, "failed": 0}
        
        for subject_id in tqdm(subject_ids, desc="Subjects"):
            print(f"\nProcessing subject {subject_id:05d}...")
            stats = self.process_subject(subject_id)
            
            total_stats["total"] += stats["total"]
            total_stats["success"] += stats["success"]
            total_stats["failed"] += stats["failed"]
            
            print(f"  Total: {stats['total']}, Success: {stats['success']}, Failed: {stats['failed']}")
        
        print("\n" + "="*50)
        print("FINAL STATISTICS:")
        print(f"  Total frames: {total_stats['total']}")
        print(f"  Successfully processed: {total_stats['success']}")
        print(f"  Failed: {total_stats['failed']}")
        print("="*50)


def get_all_subjects(root_dir: Path) -> List[int]:
    """Get all subject IDs from root directory."""
    subject_ids = []
    for d in root_dir.iterdir():
        if d.is_dir() and d.name.isdigit():
            subject_ids.append(int(d.name))
    return sorted(subject_ids)


def main():
    parser = argparse.ArgumentParser(
        description="Process CAFCA dataset with head-centered cropping and intrinsic updates"
    )
    parser.add_argument(
        '--subjects',
        type=int,
        nargs='+',
        help='Subject IDs to process (e.g., --subjects 1 30 45)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all subjects in the dataset'
    )
    parser.add_argument(
        '--root-dir',
        type=str,
        default='/home/cafca_dataset',
        help='Root directory of CAFCA dataset'
    )
    parser.add_argument(
        '--enlarge-ratio',
        type=float,
        default=1.2,
        help='Crop enlargement ratio (default: 1.15 = 15%% padding)'
    )
    parser.add_argument(
        '--output-size',
        type=int,
        default=512,
        help='Output image size (default: 512x512)'
    )
    
    args = parser.parse_args()
    
    # Determine which subjects to process
    root_dir = Path(args.root_dir)
    
    if args.all:
        subject_ids = get_all_subjects(root_dir)
        if not subject_ids:
            print(f"No subjects found in {root_dir}")
            return
        print(f"Found {len(subject_ids)} subjects to process")
    elif args.subjects:
        subject_ids = args.subjects
    else:
        parser.print_help()
        print("\nError: Must specify either --subjects or --all")
        return
    
    # Create processor
    processor = CafcaEnlargementProcessor(
        root_dir=str(root_dir),
        enlarge_ratio=args.enlarge_ratio,
        output_size=args.output_size
    )
    
    # Process subjects
    processor.process_subjects(subject_ids)


if __name__ == "__main__":
    main()

