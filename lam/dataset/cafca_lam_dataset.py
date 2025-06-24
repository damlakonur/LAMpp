# dataset/cafca_lam_dataset.py

import json
from pathlib import Path
import numpy as np
import torch

import json
from torch.utils.data import Dataset
from lam.dataset import env_paths
import numpy as np
from torchvision.io import read_image
from collections import OrderedDict

import traceback

class CafcaLamDataset(Dataset):
    def __init__(self, subject_list,
                 num_driving_frames: int = 4,
                 num_source_frames: int = 1,
                 image_size: int = 512,
                 is_val: bool = False,
                 max_tokens_in_ram: int = 128,
                 mode="lam_train"):
        """
        Dataset for loading preprocessed CAFCA data for LAM training/validation.
        Assumes images in 'masked_images' are already at target size and background handled.
        Assumes one FLAME .npz file per subject provides a single set of parameters (shape, expr, pose)
        to be used for all driving frames of that subject in an item.

        Args:
            subject_list (list): List of subject IDs (integers) to load.
            num_driving_frames (int): Number of driving frames to sample per source frame.
            num_source_frames (int): Number of source frames to sample.
            image_size (int): Expected size of images (H and W) for verification.
            is_val (bool): If True, disables certain random augmentations for validation.
            mode (str): Mode of operation.
        """
        self.root_dir = Path(env_paths.DATA_DIR)
        self.subject_list = subject_list
        self.data = []
        self.num_driving_frames = num_driving_frames
        self.num_source_frames = num_source_frames
        self.image_size = image_size
        self.is_val = is_val
        self._token_cache = OrderedDict()
        self.max_tokens_in_ram = max_tokens_in_ram
        
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
            tokens_dir = subject_base_dir / "tokens"
            img_feats_dir = subject_base_dir / "image_feats"

            if not flame_params_path.exists():
                print(f"FLAME param file not found for subject {subject_str_zfill} at {flame_params_path}. Skipping subject.")
                continue

            if not masked_images_dir.exists():
                print(f"Masked images directory not found for subject {subject_str_zfill} at {masked_images_dir}. Skipping subject.")
                continue
            
            if not masks_dir.exists():
                print(f"Foreground masks directory not found for subject {subject_str_zfill} at {masks_dir}. Skipping subject.")
                continue
            
            if not tokens_dir.exists():
                print(f"Tokens directory not found for subject {subject_str_zfill} at {tokens_dir}. Skipping subject.")
                continue
            
            if not img_feats_dir.exists():
                print(f"Image features directory not found for subject {subject_str_zfill} at {img_feats_dir}. Skipping subject.")
                continue

            camera_files = sorted(list(cameras_dir.glob("*.json")))

            if not camera_files:
                print(f"No camera files found for subject {subject_str_zfill} in {cameras_dir}. Skipping subject.")
                continue

            for cam_json_file in camera_files:
                cam_id = cam_json_file.stem # e.g., C02

                image_file = masked_images_dir / f"{cam_id}.png"
                mask_file = masks_dir / f"{cam_id}.png"
                token_file = tokens_dir / f"{cam_id}.npz"
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
                    if "world2cam" not in cam_params:
                        raise KeyError(f"'world2cam' key not found in camera file {cam_json_file}")

                    frame_data = {
                        "subject_id_int": subject_int,
                        "cam_id": cam_id,
                        "image_file_path": str(image_file),
                        "mask_file_path": str(mask_file),
                        "subject_flame_param_path": str(flame_params_path),
                        "world_2_cam_np": np.array(cam_params["world2cam"], dtype=np.float16),
                        "intrinsic_np": np.array(cam_params["K"], dtype=np.float16),
                        "token_file_path": str(token_file),
                        "is_source_candidate": cam_id in self.allowed_source_cams.get(subject_int, set()),
                    }
                    self.data.append(frame_data)
                except Exception as e:
                    print(f"Error loading data for subject {subject_str_zfill}, cam {cam_id}: {e}")

        self.subject_data = {}
        for item in self.data:
            self.subject_data.setdefault(item["subject_id_int"], []).append(item)

        self.item_list = []
        self.source_candidates = {}
        for subject_id, frames in self.subject_data.items():
            cand_idx = [i for i, fr in enumerate(frames) if fr["is_source_candidate"]]
            self.source_candidates[subject_id] = cand_idx
            for source_frame_index in range(len(frames)):
                 self.item_list.append((subject_id, source_frame_index))
        print(f"Initialized {self.__class__.__name__} with {len(self.item_list)} potential items.")
        
    def _get_token_tensor(self, path: str) -> torch.Tensor:
        """
        Returns a torch.FloatTensor(fp16) for the given token NPZ path,
        using an in-memory LRU cache to avoid duplicate loads.
        """
        cache = self._token_cache
        if path in cache:
            cache.move_to_end(path)
            print(f"Cache hit ##############.")
            return cache[path]
        print(f"Cache miss ##############. Loading into cache.")
        npz = np.load(path, mmap_mode='r')
        tensor = torch.from_numpy(npz["tokens"])

        cache[path] = tensor
        if len(cache) > self.max_tokens_in_ram:
            cache.popitem(last=False)

        return tensor
    
    def _load_image_as_tensor(self, path_str):
        img_tensor = read_image(path_str)
        if img_tensor.shape[1] != self.image_size or img_tensor.shape[2] != self.image_size:
            print(f"Warning: Image {path_str} size mismatch {img_tensor.shape[1:]} != {self.image_size}")
        return img_tensor.float() / 255.0

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

        # ------------------------------------------------------------------
        # 1. resolve primary source strictly from candidate list
        # ------------------------------------------------------------------
        subject_id, initial_idx = self.item_list[idx]
        subject_frames_info = self.subject_data[subject_id]
        candidates = self.source_candidates[subject_id]
        primary_source_idx_in_subject_list = (
            initial_idx if initial_idx in candidates
            else np.random.choice(candidates)
        )
        # ------------------------------------------------------------------
        # 2.  MULTIPLE sources — also restricted to candidates
        # ------------------------------------------------------------------
        source_frame_indices = [primary_source_idx_in_subject_list]

        if self.num_source_frames > 1:
            additional_needed = self.num_source_frames - 1
            extra_pool = [i for i in candidates if i != primary_source_idx_in_subject_list]

            if self.is_val:
                selected = extra_pool[:additional_needed]
            else:
                num_to_sample = min(additional_needed, len(extra_pool))
                selected = list(np.random.choice(extra_pool, size=num_to_sample, replace=False)) if extra_pool else []

            source_frame_indices.extend(selected)

            while len(source_frame_indices) < self.num_source_frames:
                source_frame_indices.append(source_frame_indices[-1])

        # ------------------------------------------------------------------
        # 3. driving frames: any non-source frame is eligible
        # ------------------------------------------------------------------
        candidate_driving_indices = [i for i in range(len(subject_frames_info))
                                    if i not in source_frame_indices]

        if self.num_driving_frames > 0:
            if self.is_val:
                driving_indices_in_subject_list = (candidate_driving_indices[:self.num_driving_frames]
                                                or candidate_driving_indices)
                while len(driving_indices_in_subject_list) < self.num_driving_frames:
                    driving_indices_in_subject_list.append(driving_indices_in_subject_list[-1])
            else:
                driving_indices_in_subject_list = list(np.random.choice(
                    candidate_driving_indices,
                    size=self.num_driving_frames,
                    replace=len(candidate_driving_indices) < self.num_driving_frames
                ))

        # ------------------------------------------------------------------
        # 4. load data  (unchanged apart from using new lists)
        # ------------------------------------------------------------------
        subject_flame_params = self._load_subject_flame_params(
            subject_frames_info[source_frame_indices[0]]["subject_flame_param_path"])

        source_images_list, source_cam_ids_list, source_img_tokens_list = [], [], []
        for s_idx in source_frame_indices:
            meta = subject_frames_info[s_idx]
            source_images_list.append(self._load_image_as_tensor(meta["image_file_path"]))
            # npz_token = np.load(meta["token_file_path"], mmap_mode='r')
            # source_img_tokens_list.append(torch.from_numpy(npz_token["tokens"]))
            source_img_tokens_list.append(self._get_token_tensor(meta["token_file_path"]))
            # source_img_tokens_list.append(torch.zeros((20018, 1024), dtype=torch.float16))
            source_cam_ids_list.append(meta["cam_id"])

        driving_images_list, driving_w2cs_list, driving_intrs_list, driving_cam_ids_list, driving_mask_list = [], [], [], [], []
        for d_idx in driving_indices_in_subject_list:
            meta = subject_frames_info[d_idx]
            driving_images_list.append(self._load_image_as_tensor(meta["image_file_path"]))
            driving_mask_list.append(self._load_image_as_tensor(meta["mask_file_path"]))
            driving_w2cs_list.append(torch.from_numpy(meta["world_2_cam_np"]).float())


            intr_np = meta["intrinsic_np"]
            intr_torch = torch.eye(4, dtype=torch.float16)
            intr_torch[:intr_np.shape[0], :intr_np.shape[1]] = torch.from_numpy(intr_np)
            driving_intrs_list.append(intr_torch)
            driving_cam_ids_list.append(meta["cam_id"])

        # ------------------------------------------------------------------
        # 5. assemble output dict  (unchanged)
        # ------------------------------------------------------------------
        out_item = {
            "source_rgbs": torch.stack(source_images_list),
            "tokens": torch.stack(source_img_tokens_list),
            "driving_image": torch.stack(driving_images_list),
            "driving_w2cs": torch.stack(driving_w2cs_list),
            "driving_intrs": torch.stack(driving_intrs_list),
            "driving_masks": torch.stack(driving_mask_list),
            "render_bg_colors": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).repeat(len(driving_images_list), 1),
            "uid": f"subj{subject_id}_src{''.join(source_cam_ids_list)}_drv{''.join(driving_cam_ids_list)}",
            "subject_id_int_scalar": subject_id,
            "source_cam_ids_list_scalar": source_cam_ids_list,
        }

        out_item['betas'] = subject_flame_params['betas']
        for k, v_tensor in subject_flame_params.items():
            out_item[k] = (v_tensor.unsqueeze(0).repeat(len(driving_images_list), 1)
                        if driving_images_list else torch.empty(0, *v_tensor.shape))

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
                        print(f"  Source RGBs shape: {item['source_rgbs'].shape}")
                        print(f"  Source c2ws shape: {item['source_c2ws'].shape}")
                        print(f"  Source intrs shape: {item['source_intrs'].shape}")
                        print(f"  Source FLAME betas shape: {item['betas'].shape}")
                        print(f"img _feats shape: {item['img_feats'].shape}")
                        print(f"tokens shape: {item['tokens'].shape}")

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