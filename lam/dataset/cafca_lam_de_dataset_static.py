# dataset/cafca_lam_dataset_static.py

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from lam.dataset import env_paths
from torchvision.io import read_image
from collections import OrderedDict
import itertools
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CafcaLamDataset(Dataset):
    def __init__(self, subject_list,
                 num_driving_frames: int = 4,
                 num_source_frames: int = 1,
                 image_size: int = 512,
                 is_val: bool = False,
                 max_cache_size: int = 128,
                 mode="lam_train"):
        """
        Simplified dataset for loading preprocessed CAFCA data.
        Each sample contains source and driving frames from the SAME (subject, env, expr) - just different camera views.
        """
        self.root_dir = Path(env_paths.DATA_DIR)
        self.subject_list = subject_list
        self.num_driving_frames = num_driving_frames
        self.num_source_frames = num_source_frames
        self.image_size = image_size
        self.is_val = is_val
        self._flame_cache = OrderedDict()
        self.max_cache_size = max_cache_size
        
        # Organize data by (subject, env, expr) from the start
        self.expression_data = {}
        
        for subject_int in subject_list:
            subject_str_zfill = str(subject_int).zfill(5)
            subject_dir = self.root_dir / subject_str_zfill

            if not subject_dir.exists():
                print(f"Subject directory {subject_dir} not found, skipping subject {subject_str_zfill}.")
                continue

            env_dirs = sorted([d for d in subject_dir.iterdir() if d.is_dir() and d.name.startswith('env_')])
            for env_dir in env_dirs:
                expr_dirs = sorted([d for d in env_dir.iterdir() if d.is_dir() and d.name.startswith('expr_')])

                for expr_dir in expr_dirs:
                    metadata_path = expr_dir / "metadata.json"
                    flame_params_path = expr_dir / "00400.frame"
                    cameras_dir = expr_dir / "cameras_json"  # Use processed cameras
                    
                    frames_dir = expr_dir / "frames"
                    cropped_images_dir = frames_dir / "cropped_images"  # Use processed images
                    masks_dir = frames_dir / "foreground_mask"
                    tokens_dir = frames_dir / "tokens_20k"

                    if not metadata_path.exists():
                        continue
                    
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    source_camera_ids = set(metadata.get("source_camera_ids", []))
                    
                    if not tokens_dir.exists():
                        print(f"Warning: Tokens directory not found at {tokens_dir}.")

                    camera_files = sorted(list(cameras_dir.glob("*.json")))
                    if not camera_files:
                        continue

                    # Key for this specific expression
                    expr_key = (subject_int, env_dir.name, expr_dir.name)
                    if expr_key not in self.expression_data:
                        self.expression_data[expr_key] = {
                            'frames': [],
                            'flame_params_path': str(flame_params_path)
                        }

                    for cam_json_file in camera_files:
                        cam_id = cam_json_file.stem
                        
                        image_file = cropped_images_dir / f"{cam_id}.jpg"
                        mask_file = masks_dir / f"{cam_id}.png"
                        token_file = tokens_dir / f"{cam_id}.npz"
                        
                        if not image_file.exists():
                            continue

                        try:
                            with open(cam_json_file, "r") as f:
                                cam_params = json.load(f)
                            
                            if "K" not in cam_params or "world2cam" not in cam_params:
                                continue

                            frame_data = {
                                "cam_id": cam_id,
                                "image_file_path": str(image_file),
                                "mask_file_path": str(mask_file),
                                "token_file_path": str(token_file),
                                "world_2_cam_np": np.array(cam_params["world2cam"]),
                                "intrinsic_np": np.array(cam_params["K"]),
                                "is_source_candidate": cam_id in source_camera_ids,
                            }
                            self.expression_data[expr_key]['frames'].append(frame_data)
                        except Exception as e:
                            print(f"Error loading data for subject {subject_str_zfill}, cam {cam_id}: {e}")

        # Build item list: each item is (expr_key, source_frame_indices_tuple)
        self.item_list = []
        for expr_key, expr_info in self.expression_data.items():
            frames = expr_info['frames']
            
            # Find source candidates
            source_candidates = [i for i, frame in enumerate(frames) if frame["is_source_candidate"]]
            
            if not source_candidates:
                continue
            
            # Generate all combinations of source frames
            # For each source pair, create 6 samples to get more driving view diversity
            for source_combo in itertools.combinations(source_candidates, self.num_source_frames):
                for _ in range(6):  # Create 6 samples per source pair
                    self.item_list.append((expr_key, source_combo))

        print(f"Initialized {self.__class__.__name__} with {len(self.item_list)} items across {len(self.expression_data)} expressions.")
        
    def _get_token_tensor(self, path: str) -> torch.Tensor:
        """Returns a torch.FloatTensor(fp16) for the given token NPZ path."""
        if not Path(path).exists():
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
        """Loads the preprocessed FLAME parameters from a .frame file."""
        params = torch.load(subject_flame_param_path)
        return params

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        if not (0 <= idx < len(self.item_list)):
            raise IndexError("Index out of bounds")

        # Get expression key and source frame indices
        expr_key, source_indices = self.item_list[idx]
        subject_id, env_id, expr_id = expr_key
        
        # Get all frames for this expression
        expr_info = self.expression_data[expr_key]
        frames = expr_info['frames']
        flame_params_path = expr_info['flame_params_path']
        
        # Pad source indices if needed
        source_indices = list(source_indices)
        while len(source_indices) < self.num_source_frames:
            source_indices.append(source_indices[-1])

        # Select driving frames (all frames except source frames)
        available_driving_indices = [i for i in range(len(frames)) if i not in source_indices]
        
        if not available_driving_indices:
            print(f"No driving frames available for {expr_key}")
            return None

        if self.is_val:
            chosen_driving = available_driving_indices[:self.num_driving_frames]
        else:
            num_to_sample = min(self.num_driving_frames, len(available_driving_indices))
            chosen_driving = list(np.random.choice(available_driving_indices, size=num_to_sample, replace=False))

        # Pad driving indices if needed
        while len(chosen_driving) < self.num_driving_frames:
            chosen_driving.append(chosen_driving[-1])

        # Load FLAME params (same for all frames in this expression)
        flame_params = self._load_subject_flame_params(flame_params_path)

        # Load source frame data
        source_images_list, source_cam_ids_list, source_img_tokens_list, source_w2cs_list, source_intrs_list = [], [], [], [], []
        for s_idx in source_indices:
            frame = frames[s_idx]
            source_images_list.append(self._load_image_as_tensor(frame["image_file_path"]))
            source_img_tokens_list.append(self._get_token_tensor(frame["token_file_path"]))
            source_w2cs_list.append(torch.from_numpy(frame["world_2_cam_np"]).float())
            source_intrs_list.append(torch.from_numpy(frame["intrinsic_np"]))
            source_cam_ids_list.append(frame["cam_id"])

        # Load driving frame data
        driving_images_list, driving_w2cs_list, driving_intrs_list, driving_cam_ids_list, driving_mask_list = [], [], [], [], []
        for d_idx in chosen_driving:
            frame = frames[d_idx]
            driving_images_list.append(self._load_image_as_tensor(frame["image_file_path"]))
            driving_mask_list.append(self._load_image_as_tensor(frame["mask_file_path"]))
            driving_w2cs_list.append(torch.from_numpy(frame["world_2_cam_np"]).float())
            
            intr_np = frame["intrinsic_np"]
            intr_torch = torch.eye(4, dtype=torch.float16)
            intr_torch[:intr_np.shape[0], :intr_np.shape[1]] = torch.from_numpy(intr_np)
            driving_intrs_list.append(intr_torch)
            driving_cam_ids_list.append(frame["cam_id"])

        # Assemble output dictionary
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
            "env_id": env_id,
            "exp_id": expr_id,
            "source_canon_2_cam": flame_params['canon_2_cam'].unsqueeze(0).repeat(len(source_indices), 1, 1),
            "betas": flame_params['betas'],
        }

        # FLAME parameters: expr/pose/translation are the same for all frames in this expression
        for k in ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation", "canon_2_cam"]:
            # Expand to match number of driving frames, handling any tensor dimension
            param = flame_params[k].unsqueeze(0)
            repeat_dims = [len(driving_images_list)] + [1] * flame_params[k].ndim
            out_item[k] = param.repeat(*repeat_dims)

        return out_item
    
    def get_item_by_cam_ids(self, subject_id: int, source_cam_ids: list[str], env_id: str, expr_id: str):
        """
        Return one inference item by manually specifying source cam IDs.
        Only loads source data. Driving data will be overridden later.
        
        Args:
            subject_id: Subject ID number
            source_cam_ids: List of camera IDs to use as source views
            env_id: Environment ID (e.g. 'env_0')
            expr_id: Expression ID (e.g. 'expr_0')
        """
        expr_key = (subject_id, env_id, expr_id)
        if expr_key not in self.expression_data:
            raise ValueError(f"Expression {expr_key} not found in dataset")
            
        expr_info = self.expression_data[expr_key]
        frames = expr_info['frames']
        flame_params_path = expr_info['flame_params_path']
        
        # Find frames with requested camera IDs
        cam_id_to_idx = {frame["cam_id"]: idx for idx, frame in enumerate(frames)}
        source_frame_indices = [cam_id_to_idx[cid] for cid in source_cam_ids if cid in cam_id_to_idx]

        if not source_frame_indices:
            raise ValueError(f"None of the requested camera IDs {source_cam_ids} found for {expr_key}")

        # Pad if needed
        while len(source_frame_indices) < self.num_source_frames:
            source_frame_indices.append(source_frame_indices[-1])

        # Load FLAME params
        flame_params = self._load_subject_flame_params(flame_params_path)

        # Load source frame data
        source_images_list, source_cam_ids_list, source_img_tokens_list, source_w2cs_list, source_intrs_list = [], [], [], [], []
        for s_idx in source_frame_indices:
            frame = frames[s_idx]
            source_images_list.append(self._load_image_as_tensor(frame["image_file_path"]))
            source_img_tokens_list.append(self._get_token_tensor(frame["token_file_path"]))
            source_cam_ids_list.append(frame["cam_id"])
            source_w2cs_list.append(torch.from_numpy(frame["world_2_cam_np"]).float())
            source_intrs_list.append(torch.from_numpy(frame["intrinsic_np"]))

        out_item = {
            "source_rgbs": torch.stack(source_images_list),
            "source_w2cs": torch.stack(source_w2cs_list),
            "source_intrs": torch.stack(source_intrs_list),
            "tokens": torch.stack(source_img_tokens_list),
            "driving_image": torch.empty(1, 0),
            "driving_w2cs": torch.empty(1, 0),
            "driving_intrs": torch.empty(1, 0),
            "driving_masks": torch.empty(1, 0),
            "render_bg_colors": torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
            "uid": f"subj{subject_id}_src{''.join(source_cam_ids_list)}",
            "subject_id_int_scalar": torch.tensor([subject_id]),
            "source_cam_ids_list_scalar": [source_cam_ids_list],
            "source_canon_2_cam": flame_params['canon_2_cam'].unsqueeze(0).repeat(len(source_frame_indices), 1, 1),
            "betas": flame_params['betas'].unsqueeze(0),
            "exp_id": expr_id,
            "env_id": env_id,
        }

        # Other flame params are for driving frames, which are empty in this mode
        for k in ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation", "canon_2_cam"]:
            out_item[k] = torch.empty(1, 0)

        return out_item
    

if __name__ == '__main__':
    print("="*80)
    print("DATASET DEBUG TEST")
    print("="*80)
    
    # Test with a small subset of subjects
    test_subject_ids = [1]
    
    print(f"\n1. Initializing dataset with subjects: {test_subject_ids}")
    print("-"*80)
    

    dataset = CafcaLamDataset(
        subject_list=test_subject_ids,
        num_driving_frames=4,
        num_source_frames=2,
        image_size=512,
        is_val=False,
        mode="lam_train"
    )
    
    print(f"\n2. Dataset initialized successfully!")
    print(f"   - Total items: {len(dataset)}")
    print(f"   - Total expressions: {len(dataset.expression_data)}")
    print()
    
    # Print expression details
    print("3. Expression breakdown:")
    print("-"*80)
    for expr_key, expr_info in list(dataset.expression_data.items())[:5]:
        subject_id, env_id, expr_id = expr_key
        num_frames = len(expr_info['frames'])
        source_frames = [f["cam_id"] for f in expr_info['frames'] if f["is_source_candidate"]]
        print(f"   Subject {subject_id}, {env_id}, {expr_id}:")
        print(f"      - Total frames: {num_frames}")
        print(f"      - Source candidates: {len(source_frames)} {source_frames[:3]}...")
        print(f"      - FLAME params: {Path(expr_info['flame_params_path']).name}")
    
    # Count samples per subject
    print("\n3.1 Data samples per subject:")
    print("-"*80)
    subject_item_counts = {}
    for expr_key, source_combo in dataset.item_list:
        subject_id = expr_key[0]
        subject_item_counts[subject_id] = subject_item_counts.get(subject_id, 0) + 1
    
    for subject_id in sorted(subject_item_counts.keys()):
        # Count expressions for this subject
        subject_expressions = [k for k in dataset.expression_data.keys() if k[0] == subject_id]
        print(f"   Subject {subject_id}:")
        print(f"      - Expressions: {len(subject_expressions)}")
        print(f"      - Total data samples: {subject_item_counts[subject_id]}")
    
    # Test loading a few samples
    print(f"\n4. Testing sample loading (first 3 items):")
    print("-"*80)
    
    for i in range(len(dataset)):
        print(f"\n   Sample {i}:")
        try:
            item = dataset[i]
            
            if item is None:
                print(f"      ❌ Returned None")
                continue
            
            print(f"      ✓ UID: {item['uid']}")
            # print(f"      ✓ Subject: {item['subject_id_int_scalar']}, Env: {item['env_id']}, Expr: {item['exp_id']}")
            # print(f"      ✓ Source cameras: {item['source_cam_ids_list_scalar']}")
            # print(f"      ✓ Source RGB shape: {item['source_rgbs'].shape}")
            # print(f"      ✓ Source W2C shape: {item['source_w2cs'].shape}")
            # print(f"      ✓ Source W2C: {item['source_w2cs']}")
            # print(f"      ✓ Source intrinsics shape: {item['source_intrs'].shape}")
            # print(f"      ✓ Tokens shape: {item['tokens'].shape}")
            # print(f"      ✓ Driving images shape: {item['driving_image'].shape}")
            # print(f"      ✓ Driving W2C shape: {item['driving_w2cs'].shape}")
            # print(f"      ✓ Driving intrinsics shape: {item['driving_intrs'].shape}")
            # # print(f"      ✓ Driving masks shape: {item['driving_masks'].shape}")
            # print(f"      ✓ Betas shape: {item['betas'].shape}")
            # print(f"      ✓ Expression shape: {item['expr'].shape}")
            # print(f"      ✓ Rotation shape: {item['rotation'].shape}")
            # print(f"      ✓ Translation shape: {item['translation'].shape}")
            
            # Verify data types
            assert item['source_rgbs'].dtype == torch.float32, "source_rgbs should be float32"
            assert item['tokens'].dtype == torch.float16, "tokens should be float16"
            assert 0 <= item['source_rgbs'].min() <= 1, "RGB values should be in [0,1]"
            assert 0 <= item['source_rgbs'].max() <= 1, "RGB values should be in [0,1]"
            
            print(f"      ✓ All assertions passed!")
            
        except Exception as e:
            print(f"      ❌ Error loading sample {i}: {e}")
            import traceback
            traceback.print_exc()
    
    # Test get_item_by_cam_ids method
    print(f"\n5. Testing get_item_by_cam_ids method:")
    print("-"*80)
    
    if len(dataset.expression_data) > 0:
        # Get first expression
        first_expr_key = list(dataset.expression_data.keys())[0]
        subject_id, env_id, expr_id = first_expr_key
        frames = dataset.expression_data[first_expr_key]['frames']
        
        # Get available camera IDs
        cam_ids = [f["cam_id"] for f in frames]
        
        if len(cam_ids) > 0:
            print(f"   Testing with subject={subject_id}, env={env_id}, expr={expr_id}")
            print(f"   Available cameras: {cam_ids[:5]}...")
            
            try:
                item = dataset.get_item_by_cam_ids(
                    subject_id=subject_id,
                    source_cam_ids=[cam_ids[0]],
                    env_id=env_id,
                    expr_id=expr_id
                )
                print(f"   ✓ get_item_by_cam_ids successful!")
                print(f"   ✓ UID: {item['uid']}")
                print(f"   ✓ Source cameras: {item['source_cam_ids_list_scalar']}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
                import traceback
                traceback.print_exc()
    
    print("\n" + "="*80)
    print("✓ ALL TESTS COMPLETED!")
    print("="*80)
        