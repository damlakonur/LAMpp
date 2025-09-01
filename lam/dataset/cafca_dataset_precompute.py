# dataset/cafca_lam_dataset.py

import json
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from torch.utils.data import Dataset
from lam.dataset import env_paths
import numpy as np
import cv2

import traceback
from lam.runners.infer.head_utils import img_center_padding, center_crop_according_to_mask,  calc_new_tgt_size_by_aspect

def preprocess_image(rgb_img, mask_img, pad_ratio, bg_color, 
                    aspect_standard, enlarge_ratio,
                    render_tgt_size, multiply, need_mask=True):
    rgb = np.array(rgb_img)
    rgb_raw = rgb.copy()
    if pad_ratio > 0:
        rgb = img_center_padding(rgb, pad_ratio)

    rgb = rgb / 255.0
    if need_mask:
        if rgb.shape[2] < 4:
            if mask_img is not None:
                mask = (np.array(mask_img) > 180) * 255
            if pad_ratio > 0:
                mask = img_center_padding(mask, pad_ratio)
            mask = mask / 255.0
        else:
            # rgb: [H, W, 4]
            assert rgb.shape[2] == 4
            mask = rgb[:, :, 3]   # [H, W]
    else:
        # just placeholder
        mask = np.ones_like(rgb[:, :, 0])
    if len(mask.shape) > 2:
        mask = mask[:, :, 0]

    mask = mask.astype(np.float32)
    if (rgb.shape[0] == rgb.shape[1]) and (rgb.shape[0]==512):
        rgb = cv2.resize(rgb, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_AREA)
    rgb = rgb[:, :, :3] * mask[:, :, None] + bg_color * (1 - mask[:, :, None])

    # crop image to enlarge human area.
    rgb, mask, _, _ = center_crop_according_to_mask(rgb, mask, aspect_standard, enlarge_ratio)

    # resize to render_tgt_size for training
    tgt_hw_size, _, _ = calc_new_tgt_size_by_aspect(cur_hw=rgb.shape[:2], 
                                                    aspect_standard=aspect_standard,
                                                    tgt_size=render_tgt_size, multiply=multiply)
    rgb = cv2.resize(rgb, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, dsize=(tgt_hw_size[1], tgt_hw_size[0]), interpolation=cv2.INTER_AREA)

    rgb = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    mask = torch.from_numpy(mask[:, :, None]).float().permute(2, 0, 1).unsqueeze(0)  # [1, 1, H, W]

    return rgb, mask

class CafcaDatasetPP(Dataset):
    def __init__(self, subject_list,
                 num_source_frames: int = 1, # Default to 1 source frame
                 image_size: int = 512, # Used for verification, not resizing
                 is_val: bool = False,
                 mode="lam_train"): # mode is kept for potential future use
        self.root_dir = Path(env_paths.DATA_DIR)
        self.subject_list = subject_list
        self.data = []
        self.num_source_frames = num_source_frames
        self.image_size = image_size
        self.is_val = is_val

        for subject_int in subject_list:
            subject_str_zfill = str(subject_int).zfill(5)
            subject_dir = self.root_dir / subject_str_zfill

            if not subject_dir.exists():
                print(f"Subject directory {subject_dir} not found skipping subject {subject_str_zfill}.")
                continue

            env_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith('env_')])
            for env_dir in env_dirs:
                expr_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir() and d.name.startswith('expr_')])

                for expr_dir in expr_dirs:
                    metadata_path = expr_dir / "metadata.json"
                    flame_params_path = expr_dir / "00400.frame"
                    cameras_dir = expr_dir / "cameras_json"
                    
                    frames_dir = expr_dir / "frames"
                    masked_images_dir = frames_dir / "masked_image"
                    masks_dir = frames_dir / "foreground_mask"
                    
                    if not metadata_path.exists():
                        continue
                    
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    source_camera_ids = set(metadata.get("source_camera_ids", []))
                    env_id = metadata.get("env_id", "")
                    expr_id = metadata.get("expr_id", "")

                    if not flame_params_path.exists() or not cameras_dir.exists() or \
                       not masked_images_dir.exists() or not masks_dir.exists():
                        continue
                    
                    camera_files = sorted(list(cameras_dir.glob("*.json")))
                    if not camera_files:
                        continue

                    for cam_json_file in camera_files:
                        cam_id = cam_json_file.stem
                        
                        image_file = masked_images_dir / f"{cam_id}.jpg"
                        mask_file = masks_dir / f"{cam_id}.png"
                        
                        if not image_file.exists() or not mask_file.exists():
                            continue

                        try:
                            with open(cam_json_file, "r") as f:
                                cam_params = json.load(f)
                            
                            if "K" not in cam_params:
                                raise KeyError(f"'K' key not found in camera file {cam_json_file}")
                            if "cam2world" not in cam_params:
                                raise KeyError(f"'cam2world' key not found in camera file {cam_json_file}")

                            frame_data = {
                                "subject_id_int": subject_int,
                                "cam_id": cam_id,
                                "image_file_path": str(image_file),
                                "mask_file_path": str(mask_file),
                                "subject_flame_param_path": str(flame_params_path),
                                "is_source_candidate": cam_id in source_camera_ids,
                                "env_id": env_id,
                                "expr_id": expr_id,
                            }
                            self.data.append(frame_data)
                        except Exception as e:
                            print(f"Error loading data for subject {subject_str_zfill}, cam {cam_id}: {e}")

        if not self.data:
            print(f"{self.__class__.__name__} initialized with no data items. Check paths and subject_list.")
            self.item_list = []
            return

        self.subject_data = {}
        for item in self.data:
            self.subject_data.setdefault(item["subject_id_int"], []).append(item)
        self.item_list = []
        self.source_candidates = {}
        for subject_id, frames in self.subject_data.items():
            cand_idx = [i for i, fr in enumerate(frames) if fr["is_source_candidate"]]
            self.source_candidates[subject_id] = cand_idx

            # push only the candidate indices
            for idx_in_subject in cand_idx:
                self.item_list.append((subject_id, idx_in_subject))
        if not self.item_list:
            print(f"{self.__class__.__name__} initialized with no valid items.")
        else:
            print(f"Initialized {self.__class__.__name__} with {len(self.item_list)} potential items.")




    def _load_image_as_tensor(self, path_str):
        img_pil = Image.open(path_str).convert("RGB")
        if img_pil.height != self.image_size or img_pil.width != self.image_size:
            print(f"Image {path_str} has size {img_pil.size}, but expected ({self.image_size}, {self.image_size}). Consider verifying preprocessing.")
        return TF.to_tensor(img_pil)

    def _load_subject_flame_params(self, subject_flame_param_path: str):
        """Loads the preprocessed FLAME parameters from a .frame file."""
        return torch.load(subject_flame_param_path)

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.item_list)):
            raise IndexError("Index out of bounds")

        subject_id, primary_source_idx_in_subject_list = self.item_list[idx]
        subject_frames_info = self.subject_data[subject_id]

        subject_flame_params = self._load_subject_flame_params(
            subject_frames_info[primary_source_idx_in_subject_list]["subject_flame_param_path"])

        source_frame_indices = [primary_source_idx_in_subject_list]

        source_images_list, source_cam_ids_list, source_mask_list = [], [], []
        for s_idx in source_frame_indices:
            meta = subject_frames_info[s_idx]
            rgb_img = np.array(Image.open(meta["image_file_path"]))
            mask_img = np.array(Image.open(meta["mask_file_path"]))

            rgb_tensor, mask_tensor = preprocess_image(
                rgb_img, mask_img, pad_ratio=0,
                bg_color=np.array([1.0, 1.0, 1.0]),
                aspect_standard=1.0, enlarge_ratio=[1.0, 1.0],
                render_tgt_size=512, multiply=14, need_mask=True
            )
            source_images_list.append(rgb_tensor.squeeze(0))
            source_mask_list.append(mask_tensor.squeeze(0))
            source_cam_ids_list.append(meta["cam_id"])


        out_item = {
            "source_rgbs": torch.stack(source_images_list),
            "source_masks": torch.stack(source_mask_list),
            "source_bg_colors": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).repeat(len(source_images_list), 1),
            "subject_id_int_scalar": subject_id,
            "source_cam_ids_list_scalar": source_cam_ids_list,
            "env_id": subject_frames_info[primary_source_idx_in_subject_list]["env_id"],
            "expr_id": subject_frames_info[primary_source_idx_in_subject_list]["expr_id"],
        }
        out_item['betas'] = subject_flame_params['betas']
        for k, v_tensor in subject_flame_params.items():
            # Ensure leading batch dimension of size 1; avoid repeat to keep dimensionality consistent
            if v_tensor.ndim == 1:
                out_item[k] = v_tensor.unsqueeze(0)
            else:
                out_item[k] = v_tensor

        return out_item

if __name__ == '__main__':
    if hasattr(env_paths, 'subjects_train') and env_paths.subjects_train:
        test_subjects = [1196]

        if not hasattr(env_paths, 'DATA_DIR') or not env_paths.DATA_DIR:
            print("CRITICAL: `env_paths.DATA_DIR` is not set or is empty.")
        else:
            print(f"Using CAFCA DATA_DIR: {env_paths.DATA_DIR}")
            try:
                dataset = CafcaDatasetPP(
                    subject_list=test_subjects,
                    num_source_frames=1, # Test with 1 source frame
                    image_size=512, # Expected size
                    is_val=False
                )
                print(f"Loaded {len(dataset)} items.")
                if len(dataset) > 0:
                    item = dataset[0]
                    print("\n--- Single sample summary ---")
                    for k, v in item.items():
                        if torch.is_tensor(v):
                            print(f"{k:30}: tensor {tuple(v.shape)} | dtype={v.dtype}")
                        else:
                            print(f"{k:30}: {v}")

                    print("\n--- Batch summary (batch_size=2) ---")
                    # Note: collate_fn might be needed for batching if items have different structures
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                    for batch in dataloader:
                        print(f"Batch keys: {list(batch.keys())}")
                        for k, v in batch.items():
                            if k == "source_cam_ids_list_scalar" or k == "env_id" or k == "expr_id":
                                print(f"{k:30}: {v}")

            except Exception as e:
                print(f"An error occurred during dataset initialization or testing: {e}")
                traceback.print_exc()
    else:
        print("Please define 'subjects_train' in your dataset/env_paths.py (e.g., subjects_train = [30])")