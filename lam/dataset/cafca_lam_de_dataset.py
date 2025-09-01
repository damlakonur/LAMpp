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
import itertools

import traceback

class CafcaLamDataset(Dataset):
    def __init__(self, subject_list,
                 num_driving_frames: int = 4,
                 num_source_frames: int = 1,
                 image_size: int = 512,
                 is_val: bool = False,
                 max_cache_size: int = 128,
                 mode="lam_train"):
        """
        Dataset for loading preprocessed CAFCA data for LAM training/validation.
        Assumes images in 'masked_image' are already at target size and background handled.
        Assumes one FLAME .frame file per expression provides a single set of parameters.
        Source frames are determined by metadata.json in each expression directory.
        Driving frames are selected from the same environment but a different expression.
        """
        self.root_dir = Path(env_paths.DATA_DIR)
        self.subject_list = subject_list
        self.data = []
        self.num_driving_frames = num_driving_frames
        self.num_source_frames = num_source_frames
        self.image_size = image_size
        self.is_val = is_val
        self._flame_cache = OrderedDict()
        self.max_cache_size = max_cache_size
        
        for subject_int in subject_list:
            subject_str_zfill = str(subject_int).zfill(5)
            subject_dir = self.root_dir / subject_str_zfill

            if not subject_dir.exists():
                print(f"Subject directory {subject_dir} not found – skipping subject {subject_str_zfill}.")
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
                    tokens_dir = frames_dir / "tokens"

                    if not metadata_path.exists():
                        continue
                    
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    source_camera_ids = set(metadata.get("source_camera_ids", []))

                    if not flame_params_path.exists() or not cameras_dir.exists() or \
                       not masked_images_dir.exists() or not masks_dir.exists():
                        continue
                    
                    if not tokens_dir.exists():
                        # This is a warning because user may generate tokens later
                        print(f"Warning: Tokens directory not found at {tokens_dir}.")

                    camera_files = sorted(list(cameras_dir.glob("*.json")))
                    if not camera_files:
                        continue

                    for cam_json_file in camera_files:
                        cam_id = cam_json_file.stem
                        
                        image_file = masked_images_dir / f"{cam_id}.jpg"
                        mask_file = masks_dir / f"{cam_id}.png"
                        token_file = tokens_dir / f"{cam_id}.npz"
                        
                        if not image_file.exists() or not mask_file.exists():
                            continue

                        try:
                            with open(cam_json_file, "r") as f:
                                cam_params = json.load(f)
                            
                            if "K" not in cam_params or "world2cam" not in cam_params:
                                continue

                            frame_data = {
                                "subject_id_int": subject_int,
                                "env_id": env_dir.name,
                                "expr_id": expr_dir.name,
                                "cam_id": cam_id,
                                "image_file_path": str(image_file),
                                "mask_file_path": str(mask_file),
                                "subject_flame_param_path": str(flame_params_path),
                                "world_2_cam_np": np.array(cam_params["world2cam"]),
                                "intrinsic_np": np.array(cam_params["K"]),
                                "token_file_path": str(token_file),
                                "is_source_candidate": cam_id in source_camera_ids,
                            }
                            self.data.append(frame_data)
                        except Exception as e:
                            print(f"Error loading data for subject {subject_str_zfill}, cam {cam_id}: {e}")

        # Group data by individual expression (subject, env, expr)
        expression_data = {}
        for item in self.data:
            key = (item["subject_id_int"], item["env_id"], item["expr_id"])
            if key not in expression_data:
                expression_data[key] = []
            expression_data[key].append(item)

        # Re-create subject_data for driving frame lookup
        self.subject_data = {}
        for item in self.data:
            self.subject_data.setdefault(item["subject_id_int"], []).append(item)

        # Build the item list from combinations of source frames for each expression
        self.item_list = []
        for key, frames in expression_data.items():
            # Identify source candidates for this specific expression
            source_candidates = [frame for frame in frames if frame["is_source_candidate"]]
            
            # Generate all combinations of source frames
            source_frame_combinations = itertools.combinations(source_candidates, self.num_source_frames)
            
            # Each valid combination is a data point
            self.item_list.extend(list(source_frame_combinations))

        print(f"Initialized {self.__class__.__name__} with {len(self.item_list)} potential items.")
        
    def _get_token_tensor(self, path: str) -> torch.Tensor:
        """
        Returns a torch.FloatTensor(fp16) for the given token NPZ path.
        """
        if not Path(path).exists():
            # Return a placeholder if the token file doesn't exist yet
            return torch.zeros((20018, 1024), dtype=torch.float16)

        npz = np.load(path, mmap_mode='r')
        tensor = torch.from_numpy(npz["tokens"])
        return tensor
    
    def _load_image_as_tensor(self, path_str):
        img_tensor = read_image(path_str)
        if img_tensor.shape[1] != self.image_size or img_tensor.shape[2] != self.image_size:
            print(f"Warning: Image {path_str} size mismatch {img_tensor.shape[1:]} != {self.image_size}")
        return img_tensor.float() / 255.0

    def _load_subject_flame_params(self, subject_flame_param_path: str):
        """
        Loads the preprocessed FLAME parameters from a .frame file,
        using an in-memory LRU cache to avoid duplicate loads.
        """
        # if subject_flame_param_path in self._flame_cache:
        #     self._flame_cache.move_to_end(subject_flame_param_path)
        #     return self._flame_cache[subject_flame_param_path]

        params = torch.load(subject_flame_param_path)
        
        # self._flame_cache[subject_flame_param_path] = params
        # if len(self._flame_cache) > self.max_cache_size: # Reuse same cache size limit
        #     self._flame_cache.popitem(last=False)
            
        return params

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.item_list)):
            raise IndexError("Index out of bounds")

        # item_list now stores a tuple/list of metadata dicts representing the chosen
        # source frames for this sample
        source_frames_meta = list(self.item_list[idx])

        # All source frames belong to the same subject (by construction)
        subject_id = source_frames_meta[0]["subject_id_int"]
        subject_frames_info = self.subject_data[subject_id]

        # Helper to locate the index of a given meta dict inside the subject frame list
        def _find_idx(meta_dict):
            for i, fr in enumerate(subject_frames_info):
                if (fr["cam_id"] == meta_dict["cam_id"] and
                    fr["env_id"] == meta_dict["env_id"] and
                    fr["expr_id"] == meta_dict["expr_id"]):
                    return i
            raise ValueError("Source frame not found in subject_frames_info")

        source_frame_indices = [_find_idx(m) for m in source_frames_meta]

        # Guarantee required number of sources (pad by repeating last if needed)
        while len(source_frame_indices) < self.num_source_frames:
            source_frame_indices.append(source_frame_indices[-1])

        primary_source_idx_in_subject_list = source_frame_indices[0]
        primary_source_info = subject_frames_info[primary_source_idx_in_subject_list]
        primary_env_id = primary_source_info["env_id"]
        primary_expr_id = primary_source_info["expr_id"]

        # ------------------------------------------------------------------
        # 3.1 driving frames – from *same environment* as primary source but *different expression*
        # ------------------------------------------------------------------
        candidate_driving_indices = [i for i in range(len(subject_frames_info)) if i not in source_frame_indices]

        # candidates: same env, different expr
        same_env_diff_expr_candidates = [
            i for i in candidate_driving_indices
            if subject_frames_info[i]["env_id"] == primary_env_id and 
               subject_frames_info[i]["expr_id"] != primary_expr_id
        ]

        if len(same_env_diff_expr_candidates) == 0:
            # Fallback to any other frame from the same environment if no different-expression frames are available
            same_env_diff_expr_candidates = [
                i for i in candidate_driving_indices
                if subject_frames_info[i]["env_id"] == primary_env_id
            ] or candidate_driving_indices # Ultimate fallback to any other frame

        if self.is_val:
            chosen_driving = same_env_diff_expr_candidates[: self.num_driving_frames]
        else:
            num_to_sample = min(self.num_driving_frames, len(same_env_diff_expr_candidates))
            chosen_driving = list(np.random.choice(same_env_diff_expr_candidates, size=num_to_sample, replace=False))

        # pad if fewer than required
        while len(chosen_driving) < self.num_driving_frames:
            chosen_driving.append(chosen_driving[-1])

        driving_indices_in_subject_list = chosen_driving

        # ------------------------------------------------------------------
        # 4. load data  (unchanged apart from using new lists)
        # ------------------------------------------------------------------
        # FLAME params for the *source* frame (shape/betas come from here)
        source_flame_params = self._load_subject_flame_params(
            subject_frames_info[source_frame_indices[0]]["subject_flame_param_path"])

        source_images_list, source_cam_ids_list, source_img_tokens_list, source_w2cs_list, source_intrs_list = [], [], [], [], []
        for s_idx in source_frame_indices:
            meta = subject_frames_info[s_idx]
            source_images_list.append(self._load_image_as_tensor(meta["image_file_path"]))
            source_img_tokens_list.append(self._get_token_tensor(meta["token_file_path"]))
            source_w2cs_list.append(torch.from_numpy(meta["world_2_cam_np"]).float())
            source_intrs_list.append(torch.from_numpy(meta["intrinsic_np"]))
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
            "source_w2cs": torch.stack(source_w2cs_list),
            "source_intrs": torch.stack(source_intrs_list),
            "tokens": torch.stack(source_img_tokens_list),
            "driving_image": torch.stack(driving_images_list),
            "driving_w2cs": torch.stack(driving_w2cs_list),
            "driving_intrs": torch.stack(driving_intrs_list),
            "driving_masks": torch.stack(driving_mask_list),
            "render_bg_colors": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32).repeat(len(driving_images_list), 1),
            "uid": f"subj{subject_id}_src{''.join(source_cam_ids_list)}_drv{''.join(driving_cam_ids_list)}",
            "subject_id_int_scalar": subject_id,
            "source_cam_ids_list_scalar": source_cam_ids_list,
            "source_canon_2_cam": source_flame_params['canon_2_cam'],
        }

        # ------------------------------------------------------------------
        # 6. assemble FLAME parameters
        #    - betas (shape) come from source expression (subject-specific)
        #    - expression / pose params come from the *corresponding* driving frame
        # ------------------------------------------------------------------

        out_item['betas'] = source_flame_params['betas']

        if driving_images_list:
            drive_flame_params = [
                self._load_subject_flame_params(subject_frames_info[d_idx]["subject_flame_param_path"])
                for d_idx in driving_indices_in_subject_list
            ]

            for k in ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation", "canon_2_cam"]:
                stacked = torch.stack([fp[k] for fp in drive_flame_params])
                out_item[k] = stacked
        else:
            # if no driving frames (unlikely), create empty placeholders
            for k in ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation", "canon_2_cam"]:
                out_item[k] = torch.empty(0)

        return out_item
    
    def get_item_by_cam_ids(self, subject_id: int, source_cam_ids: list[str]):
        """
        Return one inference item by manually specifying source cam IDs.
        Only loads source data. Driving data will be overridden later.
        """
        subject_frames_info = self.subject_data[subject_id]

        # Dynamic candidate indices (source eligible)
        candidates = [i for i, fr in enumerate(subject_frames_info) if fr["is_source_candidate"]]

        # Helper to locate the index of a given meta dict inside the subject frame list
        cam_id_to_idx = {frame_info["cam_id"]: idx for idx, frame_info in enumerate(subject_frames_info)}
        source_frame_indices = [cam_id_to_idx[cid] for cid in source_cam_ids if cid in cam_id_to_idx]

        if not source_frame_indices:
            return None

        subject_flame_params = self._load_subject_flame_params(
            subject_frames_info[source_frame_indices[0]]['subject_flame_param_path']
        )

        source_images_list, source_cam_ids_list, source_img_tokens_list, source_w2cs_list, source_intrs_list = [], [], [], [], []
        for s_idx in source_frame_indices:
            meta = subject_frames_info[s_idx]
            source_images_list.append(self._load_image_as_tensor(meta["image_file_path"]))
            source_img_tokens_list.append(self._get_token_tensor(meta["token_file_path"]))
            source_cam_ids_list.append(meta["cam_id"])
            source_w2cs_list.append(torch.from_numpy(meta["world_2_cam_np"]).float())
            source_intrs_list.append(torch.from_numpy(meta["intrinsic_np"]))
        out_item = {
            "source_rgbs": torch.stack(source_images_list).unsqueeze(0),
            "source_w2cs": torch.stack(source_w2cs_list).unsqueeze(0),
            "source_intrs": torch.stack(source_intrs_list).unsqueeze(0),
            "tokens": torch.stack(source_img_tokens_list).unsqueeze(0),
            "driving_image": torch.empty(1, 0),
            "driving_w2cs": torch.empty(1, 0),
            "driving_intrs": torch.empty(1, 0),
            "driving_masks": torch.empty(1, 0),
            "render_bg_colors": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            "uid": f"subj{subject_id}_src{''.join(source_cam_ids_list)}",
            "subject_id_int_scalar": torch.tensor([subject_id]),
            "source_cam_ids_list_scalar": [source_cam_ids_list],
            "source_canon_2_cam": subject_flame_params['canon_2_cam'].unsqueeze(0),
        }

        out_item['betas'] = subject_flame_params['betas'].unsqueeze(0)
        # Other flame params are for driving frames, which are empty in this mode
        for k in ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation", "canon_2_cam"]:
             out_item[k] = torch.empty(1, 0)

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
                    print("\n--- Single sample summary ---")
                    for k, v in item.items():
                        if torch.is_tensor(v):
                            print(f"{k:15}: tensor {tuple(v.shape)} | dtype={v.dtype}")
                        else:
                            print(f"{k:15}: {v}")

                    print("\nQuick access:")
                    print(f"  UID              : {item['uid']}")
                    print(f"  #source frames   : {item['source_rgbs'].shape[0]}")
                    print(f"  #driving frames  : {item['driving_image'].shape[0]}")

                    # Sanity-check key names
                    print(f"  Source W2Cs shape: {item['source_w2cs'].shape}")
                    print(f"  Driving W2Cs shape: {item['driving_w2cs'].shape}")
                    print(f"  Source canon_2_cam shape: {item['source_canon_2_cam'].shape}")
                    print(f"  Driving canon_2_cam shape: {item['canon_2_cam'].shape}")

                    print("\nExample of a batch returned by DataLoader:")
                    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
                    batch = next(iter(dataloader))
                    print(f"Batch keys: {batch.keys()}")
                    print(f"Batch['source_rgbs'] shape  : {batch['source_rgbs'].shape}")
                    print(f"Batch['betas'] shape        : {batch['betas'].shape}")
                    print(f"Batch['driving_image'] shape: {batch['driving_image'].shape}")
                    print(f"Batch['expr'] shape         : {batch['expr'].shape}")
                    print(f"Batch UID                   : {batch['uid']}")
                    print(f"Batch source W2Cs           : {batch['source_w2cs'].shape}")
                    print(f"Batch driving W2Cs          : {batch['driving_w2cs'].shape}")
                    print(f"Batch source intrs          : {batch['source_intrs'].shape}")
                    print(f"Batch driving intrs         : {batch['driving_intrs'].shape}")
                    print(f"Batch source canon_2_cam    : {batch['source_canon_2_cam'].shape}")
                    print(f"Batch driving canon_2_cam   : {batch['canon_2_cam'].shape}")
                    

            except Exception as e:
                print(f"An error occurred during dataset initialization or testing: {e}")
                traceback.print_exc()
    else:
        print("Please define 'subjects_train' in your dataset/env_paths.py (e.g., subjects_train = [30])")