# offline_preprocess_cafca_stage1.py

import os
import cv2
import numpy as np
from PIL import Image
import json
from pathlib import Path
import traceback
import time
from torch.utils.data import Dataset
from tools.flame_tracking_single_image import FlameTrackingSingleImage, expand_bbox
from lam.runners.infer.lam import parse_configs
import cv2
import numpy as np
import torch
import torchvision
import tyro
import yaml
import pyvista as pv
from loguru import logger
from PIL import Image
from dreifus.pyvista import render_from_camera
from dreifus.camera import CameraCoordinateConvention, PoseType
from dreifus.matrix import Intrinsics, Pose

# --- Configuration for Preprocessing ---
CROP_PADDING_SCALE = 1.65  # From FlameTrackingSingleImage.preprocess expand_bbox scale
INTERMEDIATE_IMAGE_SIZE = 1024 # Resize target in FlameTrackingSingleImage.preprocess
ERROR_CODE = {'FailedToDetect': 1, 'FailedToOptimize': 2, 'FailedToExport': 3}

from lam.dataset import env_paths

class CafcaDataset(Dataset):
    def __init__(self, subject_list):
        self.cfg = parse_configs()
        self.root_dir = Path(env_paths.DATA_DIR)
        self.subject_list = subject_list
        self.data = []
        self.flametracking = FlameTrackingSingleImage(output_dir='tracking_output',
                                             alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
                                             vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
                                             human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
                                             facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
                                             detect_iris_landmarks=True,
                                             args = self.cfg)

        for subject_int in subject_list:
            subject_str_zfill = str(subject_int).zfill(5)
            subject_dir_actual = self.get_subject_dir(subject_int)
            
            cameras_dir = subject_dir_actual / "cameras_json"
            images_dir = subject_dir_actual / "masked_images"
            
            camera_files = sorted(list(cameras_dir.glob("*.json")))
            image_files = sorted(list(images_dir.glob("*.png")))

            if not camera_files:
                print(f"Warning: No camera files found for subject {subject_int} in {cameras_dir}")
                continue
            if not image_files:
                print(f"Warning: No image files found for subject {subject_int} in {images_dir}")
                continue
            
            if len(camera_files) != len(image_files):
                # Using your original error handling for mismatch
                raise ValueError(
                    f"Mismatch between cameras ({len(camera_files)}) and images ({len(image_files)}) for subject {subject_int}"
                )

            for cam_file, img_file in zip(camera_files, image_files):
                # Your original camera_id logic
                camera_id = cam_file.stem.split(".")[-1] 
                if "." in cam_file.stem: 
                    camera_id = cam_file.stem.split(".")[-1]
                else:
                    camera_id = cam_file.stem
                try:
                    with open(cam_file, "r") as f:
                        cam_params = json.load(f)

                    if "cam2world" not in cam_params:
                        raise KeyError(f"'cam2world' key not found in {cam_file}")
                    if "K" not in cam_params:
                        raise KeyError(f"'K' key not found in {cam_file}")

                    self.data.append(
                        {
                            "subject": subject_int,
                            "subject_id_str": subject_str_zfill,
                            "cam_2_world": cam_params["cam2world"],
                            "intrinsic": np.array(cam_params["K"]),
                            "image_file_path": str(img_file), # Path to image in "masked_images"
                            "cam_id": camera_id,
                            "original_cam_json_path": str(cam_file),
                            "world_2_cam": cam_params["world2cam"]
                        }
                    )
                except Exception as e:
                    print(f"Error loading data for {cam_file} or {img_file}: {e}")
        
        if not self.data:
            print("Warning: CafcaDataset initialized with no data items.")

    def get_subject_dir(self, subject_id_int): # Takes int
        subject_str_zfill = str(subject_id_int).zfill(5)
        return (
            self.root_dir
            / subject_str_zfill 
            / f"{env_paths.EXPRESSION_ID}_{env_paths.ENVIRONMENT_ID}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return item
    
    def preprocess(self, input_image_path: str,
                original_intrinsics_np: np.ndarray):

        if not os.path.exists(input_image_path):
            logger.warning(f'{input_image_path} does not exist!')
            return ERROR_CODE['FailedToDetect']

        start_time = time.time()
        logger.info('Starting preprocessing…')

        # ────────────────────────────────────────────────────────────
        # 1. Face / bbox detection
        # ────────────────────────────────────────────────────────────
        frame = torchvision.io.read_image(input_image_path)[:3, ...]      # C,H,W, 0-255 uint8
        try:
            _, frame_bbox, _ = self.flametracking.vgghead_encoder(frame, 0)
        except Exception:
            logger.error('Failed to detect face')
            return ERROR_CODE['FailedToDetect']

        if frame_bbox is None:
            logger.error('Failed to detect face')
            return ERROR_CODE['FailedToDetect']

        # expand and cast to int
        frame_bbox = expand_bbox(frame_bbox, scale=1.65).long()           # (x1,y1,x2,y2)

        # ────────────────────────────────────────────────────────────
        # 2. Crop & resize to 1024×1024
        # ────────────────────────────────────────────────────────────
        x1, y1, x2, y2 = map(int, frame_bbox)
        cropped_frame = torchvision.transforms.functional.crop(
            frame, top=y1, left=x1, height=y2 - y1, width=x2 - x1)
        cropped_frame = torchvision.transforms.functional.resize(
            cropped_frame, (1024, 1024), antialias=True)

        # ────────────────────────────────────────────────────────────
        # 3. Matting
        # ────────────────────────────────────────────────────────────
        cropped_frame, mask = self.flametracking.matting_engine(
            cropped_frame / 255.0, return_type='matting', background_rgb=1.0)
        cropped_frame = cropped_frame.cpu() * 255.0

        saved_image = np.round(
            cropped_frame.permute(1, 2, 0).numpy()).astype(np.uint8)[:, :, (2, 1, 0)]
        mask = np.array(mask.cpu() * 255.0, dtype=np.uint8)

        # ────────────────────────────────────────────────────────────
        # 4. UPDATE INTRINSICS
        # ────────────────────────────────────────────────────────────
        # original_intrinsics_np is expected in the form:
        # [[fx,  0, cx],
        #  [ 0, fy, cy],
        #  [ 0,  0,  1]]
        updated_intrinsics_np = original_intrinsics_np.copy()

        # translate because we shifted the optical centre by cropping
        updated_intrinsics_np[0, 2] -= x1     # cx' = cx - left
        updated_intrinsics_np[1, 2] -= y1     # cy' = cy - top

        # scale because we resized the cropped patch to 1024×1024
        crop_w, crop_h = x2 - x1, y2 - y1
        sx, sy = 1024.0 / crop_w, 1024.0 / crop_h
        updated_intrinsics_np[0, 0] *= sx     # fx'
        updated_intrinsics_np[1, 1] *= sy     # fy'
        updated_intrinsics_np[0, 2] *= sx     # cx''
        updated_intrinsics_np[1, 2] *= sy     # cy''

        # ────────────────────────────────────────────────────────────
        end_time = time.time()
        torch.cuda.empty_cache()
        logger.info(f'Finished preprocessing. Time: {end_time - start_time:.2f}s')

        return saved_image, mask, updated_intrinsics_np

# --- Helper Functions for Preprocessing (Identical to previous correct version) ---
def preprocess_image_and_intrinsics(
    original_image_np, 
    binary_foreground_mask_np, 
    original_intrinsics_np
):
    """
    Applies Stage 1 preprocessing: RoI from mask -> Crop with Padding -> Resize to 1024x1024 -> Adjust Intrinsics.
    """
    # 1. RoI Definition (using the bounding box of the provided binary foreground mask)
    ys, xs = np.where(binary_foreground_mask_np > 128) 
    if len(xs) == 0 or len(ys) == 0:
        print(f"Warning: Empty or near-empty binary foreground mask. Using full image as RoI.")
        x_min_roi, y_min_roi, x_max_roi, y_max_roi = 0, 0, original_image_np.shape[1]-1, original_image_np.shape[0]-1
    else:
        x_min_roi, y_min_roi = np.min(xs), np.min(ys)
        x_max_roi, y_max_roi = np.max(xs), np.max(ys)

    # 2. Expand Bounding Box
    center_x_roi, center_y_roi = (x_min_roi + x_max_roi) / 2, (y_min_roi + y_max_roi) / 2
    height_roi, width_roi = y_max_roi - y_min_roi, x_max_roi - x_min_roi
    
    height_roi = max(1, height_roi) 
    width_roi = max(1, width_roi)   
    
    extension_size = max(height_roi, width_roi) * CROP_PADDING_SCALE
    
    crop_x_start = int(center_x_roi - extension_size / 2)
    crop_y_start = int(center_y_roi - extension_size / 2)
    crop_x_end = int(center_x_roi + extension_size / 2) 
    crop_y_end = int(center_y_roi + extension_size / 2) 

    crop_x_start = max(0, crop_x_start)
    crop_y_start = max(0, crop_y_start)
    crop_x_end = min(original_image_np.shape[1], crop_x_end)
    crop_y_end = min(original_image_np.shape[0], crop_y_end)

    # 3. Crop the original image and the binary foreground mask
    cropped_image_np = original_image_np[crop_y_start:crop_y_end, crop_x_start:crop_x_end, :]
    cropped_binary_mask_np = binary_foreground_mask_np[crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    if cropped_image_np.shape[0] == 0 or cropped_image_np.shape[1] == 0:
        print(f"Warning: Crop resulted in an empty image. Using original image and mask for resize.")
        cropped_image_np = original_image_np.copy()
        cropped_binary_mask_np = binary_foreground_mask_np.copy()
        crop_x_start, crop_y_start = 0, 0 
        crop_x_end, crop_y_end = original_image_np.shape[1], original_image_np.shape[0]

    # 4. Resize cropped image and mask to INTERMEDIATE_IMAGE_SIZE (1024x1024)
    resized_image_np = cv2.resize(cropped_image_np, (INTERMEDIATE_IMAGE_SIZE, INTERMEDIATE_IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    resized_binary_mask_np = cv2.resize(cropped_binary_mask_np, (INTERMEDIATE_IMAGE_SIZE, INTERMEDIATE_IMAGE_SIZE), interpolation=cv2.INTER_NEAREST)
    
    # 5. Adjust Camera Intrinsics
    adjusted_intrinsics_np = original_intrinsics_np.copy()
    adjusted_intrinsics_np[0, 2] -= crop_x_start 
    adjusted_intrinsics_np[1, 2] -= crop_y_start  

    actual_crop_width = crop_x_end - crop_x_start
    actual_crop_height = crop_y_end - crop_y_start

    if actual_crop_width <= 0 or actual_crop_height <= 0:
        print(f"Warning: Crop dimensions are invalid ({actual_crop_width}x{actual_crop_height}). Using 1.0 for intrinsic scaling.")
        scale_x_resize, scale_y_resize = 1.0, 1.0
    else:
        scale_x_resize = INTERMEDIATE_IMAGE_SIZE / actual_crop_width
        scale_y_resize = INTERMEDIATE_IMAGE_SIZE / actual_crop_height
    
    adjusted_intrinsics_np[0,0] *= scale_x_resize 
    adjusted_intrinsics_np[1,1] *= scale_y_resize 
    adjusted_intrinsics_np[0,2] *= scale_x_resize 
    adjusted_intrinsics_np[1,2] *= scale_y_resize 
        
    return resized_image_np, resized_binary_mask_np, adjusted_intrinsics_np

# --- Main Script ---
if __name__ == "__main__":
    ARG_BINARY_MASKS_DIRNAME = "foreground_mask"
    OUTPUT_PREPROCESSED_SUBDIR_NAME = "preprocessed_1024"

    # --- Initialize CafcaDataset ---
    if not hasattr(env_paths, 'subjects_train') or not env_paths.subjects_train:
        print("Error: env_paths.subjects_train is not defined or is empty. Please set it in your env_paths.py.")
        print("Example: subjects_train = [30]")
        exit()
        
    dataset_instance = CafcaDataset([30])
    breakpoint()

    if len(dataset_instance) == 0:
        print("CafcaDataset is empty. Please check paths and subject list in env_paths.py. Exiting.")
    else:
        print(f"CafcaDataset loaded with {len(dataset_instance)} items.")

    # --- Iterate through dataset items and preprocess ---
    for i in range(len(dataset_instance)):
        item_data = None 
        try:
            # Use the __getitem__ method of your CafcaDataset
            item_data = dataset_instance[i] 
            
            subject_id_str = item_data["subject_id_str"]
            cam_id = item_data["cam_id"]
            original_image_path_str = item_data["image_file_path"]
            original_k_np = item_data["intrinsic"] 
            original_cam_json_path_str = item_data["original_cam_json_path"]

            print(f"\nProcessing Subject: {subject_id_str}, Camera ID: {cam_id}")
            print(f"  Input Image: {original_image_path_str}")

            original_image_path = Path(original_image_path_str)
            subject_base_dir = original_image_path.parent.parent 
            
            binary_mask_path = subject_base_dir / ARG_BINARY_MASKS_DIRNAME / original_image_path.name

            output_base_dir = subject_base_dir / OUTPUT_PREPROCESSED_SUBDIR_NAME
            output_processed_images_dir = output_base_dir / "images"
            output_processed_masks_dir = output_base_dir / "masks"
            output_processed_cameras_dir = output_base_dir / "cameras_json"

            os.makedirs(output_processed_images_dir, exist_ok=True)
            os.makedirs(output_processed_masks_dir, exist_ok=True)
            os.makedirs(output_processed_cameras_dir, exist_ok=True)

            if not binary_mask_path.exists():
                print(f"  Binary foreground mask not found: {binary_mask_path}, skipping.")
                continue
            
            original_image_pil = Image.open(original_image_path).convert("RGB")
            original_image_np = np.array(original_image_pil)

            binary_mask_pil = Image.open(binary_mask_path).convert("L")
            binary_mask_np = np.array(binary_mask_pil)

            processed_img_np, processed_mask_np, adjusted_k_np = dataset_instance.preprocess(
                original_image_path,
                original_k_np 
            )
            image_save_path = output_processed_images_dir / f"{cam_id}.png"
            # cv2.imwrite(str(image_save_path), original_image_pil)
            # Image.fromarray(original_image_np).save(image_save_path)
            Image.fromarray(processed_img_np).save(output_processed_images_dir / f"{cam_id}.png")
            Image.fromarray(processed_mask_np, mode='L').save(output_processed_masks_dir / f"{cam_id}.png")

            with open(Path(original_cam_json_path_str), 'r') as f_orig_cam:
                original_full_cam_params = json.load(f_orig_cam)

            new_cam_params_to_save = original_full_cam_params.copy() 
            new_cam_params_to_save["K"] = original_k_np.tolist()
            new_cam_params_to_save["cam2world"] = item_data["cam_2_world"]
            new_cam_params_to_save["world2cam"] = item_data["world_2_cam"]
            new_cam_params_to_save["height"] = processed_img_np.shape[0] 
            new_cam_params_to_save["width"] = processed_img_np.shape[1]  
            new_cam_params_to_save["original_height_before_stage1"] = original_image_np.shape[0]
            new_cam_params_to_save["original_width_before_stage1"] = original_image_np.shape[1]

            with open(output_processed_cameras_dir / f"{cam_id}.json", 'w') as f:
                json.dump(new_cam_params_to_save, f, indent=4)
            
            print(f"  Successfully processed and saved to {output_base_dir}")

        except Exception as e:
            img_path_for_error = item_data.get('image_file_path', 'N/A') if item_data else 'N/A'
            print(f"  Error processing item {i} (Image: {img_path_for_error}): {e}")
            traceback.print_exc()

    print("\nOffline preprocessing (Stage 1) finished.")

