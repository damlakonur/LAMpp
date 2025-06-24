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
                 num_driving_frames: int = 4,
                 num_source_frames: int = 1, # Default to 1 source frame
                 image_size: int = 512, # Used for verification, not resizing
                 is_val: bool = False,
                 mode="lam_train"): # mode is kept for potential future use
        self.root_dir = Path(env_paths.DATA_DIR)
        self.subject_list = subject_list
        self.data = []
        self.num_driving_frames = num_driving_frames
        self.num_source_frames = num_source_frames
        self.image_size = image_size
        self.is_val = is_val
        source_json_path = self.root_dir / "available_source_views.json"
        with open(source_json_path, "r") as f:
            raw = json.load(f)
        self.allowed_source_cams = {
            int(entry["subject_id"]): set(entry["cameras_with_non_zero_lmks"])
            for entry in raw
        }

        for subject_int in subject_list:
            subject_str_zfill = str(subject_int).zfill(5)
            subject_base_dir = (
                self.root_dir
                / subject_str_zfill
                / f"{env_paths.EXPRESSION_ID}_{env_paths.ENVIRONMENT_ID}"
            )

            flame_params_path = subject_base_dir / f"{subject_str_zfill}{env_paths.FLAME_FILENAME}"

            masked_images_dir = subject_base_dir / "masked_images"
            cameras_dir = subject_base_dir / "cameras_json"
            masks_dir = subject_base_dir / "foreground_mask"

            if not flame_params_path.exists():
                print(f"FLAME param file not found for subject {subject_str_zfill} at {flame_params_path}. Skipping subject.")
                continue

            if not masked_images_dir.exists():
                print(f"Masked images directory not found for subject {subject_str_zfill} at {masked_images_dir}. Skipping subject.")
                continue
            
            if not masks_dir.exists():
                print(f"Foreground masks directory not found for subject {subject_str_zfill} at {masks_dir}. Skipping subject.")
                continue
            

            camera_files = sorted(list(cameras_dir.glob("*.json")))

            if not camera_files: # Should not happen if masked_images exist for those cameras
                print(f"No camera files found for subject {subject_str_zfill} in {cameras_dir}. Skipping subject.")
                continue

            for cam_json_file in camera_files:
                cam_id = cam_json_file.stem # e.g., C02

                image_file = masked_images_dir / f"{cam_id}.png"
                mask_file = masks_dir / f"{cam_id}.png"
                if not mask_file.exists():
                    print(f"Mask file {mask_file} not found for subject {subject_str_zfill}, cam {cam_id}. Skipping item.")
                    continue

                if not image_file.exists():
                    # This might happen if camera_json exists but corresponding image doesn't
                    print(f"Image {image_file} not found for subject {subject_str_zfill}, cam {cam_id}, though camera file exists. Skipping item.")
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
                        "is_source_candidate": cam_id in self.allowed_source_cams.get(subject_int, set()),
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
        """Loads the single set of FLAME parameters from the subject's .npz file."""
        flame_param = dict(np.load(subject_flame_param_path), allow_pickle=True)

        flame_param_tensor = {}
        flame_param_tensor['expr'] = torch.FloatTensor(flame_param['expr'])[0]
        flame_param_tensor['rotation'] = torch.FloatTensor(flame_param['rotation'])[0]
        flame_param_tensor['neck_pose'] = torch.FloatTensor(flame_param['neck_pose'])[0]
        flame_param_tensor['jaw_pose'] = torch.FloatTensor(flame_param['jaw_pose'])[0]
        flame_param_tensor['eyes_pose'] = torch.FloatTensor(flame_param['eyes_pose'])[0]
        flame_param_tensor['translation'] = torch.FloatTensor(flame_param['translation'])[0]
        flame_param_tensor['betas'] = torch.FloatTensor(flame_param['shape'])   
        return flame_param_tensor

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
        }

        out_item['betas'] = subject_flame_params['betas']
        for k, v_tensor in subject_flame_params.items():
            out_item[k] = v_tensor.unsqueeze(0).repeat(4, 1)

        return out_item

if __name__ == '__main__':
    if hasattr(env_paths, 'subjects_train') and env_paths.subjects_train:
        test_subjects = env_paths.subjects_train[:1]

        if not hasattr(env_paths, 'DATA_DIR') or not env_paths.DATA_DIR:
            print("CRITICAL: `env_paths.DATA_DIR` is not set or is empty.")
        else:
            print(f"Using CAFCA DATA_DIR: {env_paths.DATA_DIR}")
            try:
                dataset = CafcaLamDataset(
                    subject_list=test_subjects,
                    num_driving_frames=2,
                    num_source_frames=2, # Test with 2 source frames
                    image_size=512, # Expected size
                    is_val=False
                )
                print(f"Loaded {len(dataset)} items.")
                if len(dataset) > 0:
                    item = dataset[0]
                    print("\nExample of a single item returned by __getitem__ (item.keys()):")
                    print(sorted(item.keys()))

                    print(f"\nUID: {item['uid']}")
                    print(f"Number of source frames: {item['source_rgbs'].shape[0] if item['source_rgbs'].nelement() > 0 else 0}")
                    if item['source_rgbs'].nelement() > 0:
                        breakpoint()
                        print(f"  Source RGBs shape: {item['source_rgbs'].shape}")
                        print(f"  Source c2ws shape: {item['source_c2ws'].shape}")
                        print(f"  Source intrs shape: {item['source_intrs'].shape}")
                        print(f"  Source FLAME betas shape: {item['betas'].shape}")

                    print(f"Number of driving frames: {item['driving_image'].shape[0] if item['driving_image'].nelement() > 0 else 0}")
                    if item['driving_image'].nelement() > 0:
                        print(f"  Driving RGBs (driving_image) shape: {item['driving_image'].shape}")
                        print(f"  Driving c2ws shape: {item['driving_c2ws'].shape}")
                        print(f"  Driving intrs shape: {item['driving_intrs'].shape}")
                        print(f"  Driving FLAME expr shape: {item['expr'].shape}")

                    print("\nExample of a batch returned by DataLoader:")
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                    batch = next(iter(dataloader))
                    print(f"Batch keys: {batch.keys()}")
                    print(f"Batch['source_rgbs'] shape: {batch['source_rgbs'].shape}") # Expected: [Batch, NumSource, C, H, W]
                    print(f"Batch['source_betas'] shape: {batch['betas'].shape}") # Expected: [Batch, N_shape]
                    print(f"Batch['driving_image'] shape: {batch['driving_image'].shape}")# Expected: [Batch, NumDriving, C, H, W]
                    print(f"Batch['expr'] shape: {batch['expr'].shape}") # Expected: [Batch, NumDriving, N_expr]
                    print(f"uuid: {batch['uid']}")
                    print(f"Batch['source_c2ws']: {batch['source_c2ws']}")
                    print(f"Batch['driving_c2ws']: {batch['driving_c2ws']}")
                    print(f"Batch['source_intrs']: {batch['source_intrs']}")
                    print(f"Batch['driving_intrs']: {batch['driving_intrs']}")
                    

            except Exception as e:
                print(f"An error occurred during dataset initialization or testing: {e}")
                traceback.print_exc()
    else:
        print("Please define 'subjects_train' in your dataset/env_paths.py (e.g., subjects_train = [30])")