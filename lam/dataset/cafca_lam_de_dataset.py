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
                    # masked_images_dir = frames_dir / "masked_image"
                    cropped_images_dir = frames_dir / "cropped_images"
                    masks_dir = frames_dir / "foreground_mask"
                    tokens_dir = frames_dir / "tokens_20k"
                    image_feats_dir = frames_dir / "image_feats"

                    if not metadata_path.exists():
                        continue
                    
                    with open(metadata_path, "r") as f:
                        metadata = json.load(f)
                    source_camera_ids = set(metadata.get("source_camera_ids", []))

                    # if not flame_params_path.exists() or not cameras_dir.exists() or \
                    #    not masked_images_dir.exists() or not masks_dir.exists():
                    #     continue
                    
                    if not tokens_dir.exists():
                        # This is a warning because user may generate tokens later
                        print(f"Warning: Tokens directory not found at {tokens_dir}.")
                        
                    # if not image_feats_dir.exists():
                    #     # This is a warning because user may generate image feats later
                    #     print(f"Warning: Image feats directory not found at {image_feats_dir}.")

                    camera_files = sorted(list(cameras_dir.glob("*.json")))
                    if not camera_files:
                        continue

                    for cam_json_file in camera_files:
                        cam_id = cam_json_file.stem
                        
                        image_file = cropped_images_dir / f"{cam_id}.jpg"
                        mask_file = masks_dir / f"{cam_id}.png"
                        token_file = tokens_dir / f"{cam_id}.npz"
                        image_feat_file = image_feats_dir / f"{cam_id}.npz"
                        
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
                                "image_feat_file_path": str(image_feat_file),
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
    
    def _get_image_feat_tensor(self, path: str) -> torch.Tensor:
        """
        Returns a torch.FloatTensor(fp16) for the given image feat NPZ path.
        """
        npz = np.load(path, mmap_mode='r')
        tensor = torch.from_numpy(npz["image_feats"])
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
        # set jaw_pose, neck_pose, eyes_pose to identity
        # params["eyes_pose"] = torch.zeros_like(params["eyes_pose"])
        
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
        if not same_env_diff_expr_candidates:
            print(f"No driving frames found for subject {subject_id}, env {primary_env_id}, expr {primary_expr_id}")
            return None

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

        source_images_list, source_cam_ids_list, source_img_tokens_list, source_w2cs_list, source_intrs_list, source_img_feats_list = [], [], [], [], [], []
        for s_idx in source_frame_indices:
            meta = subject_frames_info[s_idx]
            source_images_list.append(self._load_image_as_tensor(meta["image_file_path"]))
            source_img_tokens_list.append(self._get_token_tensor(meta["token_file_path"]))
            source_img_feats_list.append(self._get_image_feat_tensor(meta["image_feat_file_path"]))
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
            "source_img_feats": torch.stack(source_img_feats_list),
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
        if subject_id not in self.subject_data:
            raise ValueError(f"Subject {subject_id} not found in dataset")
            
        subject_frames_info = self.subject_data[subject_id]
        
        # Filter frames for the specified environment and expression
        frames = [f for f in subject_frames_info if f["env_id"] == env_id and f["expr_id"] == expr_id]
        
        if not frames:
            raise ValueError(f"No frames found for subject {subject_id}, env {env_id}, expr {expr_id}")

        # Helper to locate the index of a given meta dict inside the filtered frames
        cam_id_to_idx = {frame["cam_id"]: idx for idx, frame in enumerate(frames)}
        source_frame_indices = [cam_id_to_idx[cid] for cid in source_cam_ids if cid in cam_id_to_idx]

        if not source_frame_indices:
            raise ValueError(f"None of the requested camera IDs {source_cam_ids} found for subject {subject_id}")

        # Guarantee required number of sources (pad by repeating last if needed)
        while len(source_frame_indices) < self.num_source_frames:
            source_frame_indices.append(source_frame_indices[-1])

        # Load FLAME params from first source frame
        source_flame_params = self._load_subject_flame_params(
            frames[source_frame_indices[0]]["subject_flame_param_path"]
        )

        source_images_list, source_cam_ids_list, source_img_tokens_list, source_w2cs_list, source_intrs_list = [], [], [], [], []
        for s_idx in source_frame_indices:
            meta = frames[s_idx]
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
            "source_canon_2_cam": source_flame_params['canon_2_cam'].unsqueeze(0),
            "betas": source_flame_params['betas'].unsqueeze(0)
        }

        # Other flame params are for driving frames, which are empty in this mode
        for k in ["expr", "rotation", "neck_pose", "jaw_pose", "eyes_pose", "translation", "canon_2_cam"]:
            out_item[k] = torch.empty(1, 0)

        return out_item
    

if __name__ == '__main__':
    cafca_subject_ids_train: [0, 1, 2, 4, 5, 10, 11, 12, 13, 15, 17, 18, 19, 20, 21, 24, 25, 27, 28, 30, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 44, 45, 46, 47, 48, 49, 50, 51, 54, 55, 56, 57, 58, 60, 63, 66, 68, 69, 70, 71, 72, 74, 75, 76, 77, 78, 79, 80, 82, 83, 84, 86, 87, 88, 89, 90, 91, 93, 94, 95, 96, 98, 99, 100, 101, 103, 104, 108, 109, 110, 111, 112, 113, 114, 115, 116, 118, 119, 120, 121, 124, 127, 129, 130, 132, 133, 134, 135, 136, 137, 138, 139, 142, 143, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 157, 160, 161, 162, 165, 166, 167, 169, 170, 171, 172, 173, 174, 175, 177, 178, 179, 181, 182, 183, 184, 186, 187, 188, 189, 190, 193, 194, 195, 196, 198, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 212, 215, 217, 218, 219, 220, 222, 226, 228, 229, 230, 232, 233, 235, 238, 239, 240, 241, 242, 245, 246, 247, 248, 249, 251, 253, 254, 255, 257, 259, 260, 261, 262, 263, 264, 266, 267, 268, 272, 273, 275, 277, 278, 280, 281, 283, 284, 285, 286, 287, 289, 290, 291, 293, 294, 295, 296, 297, 299, 300, 301, 302, 303, 307, 309, 312, 313, 314, 317, 318, 319, 320, 322, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 339, 340, 342, 343, 344, 345, 346, 347, 348, 350, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 364, 365, 366, 368, 370, 373, 374, 375, 377, 379, 380, 383, 384, 385, 386, 387, 388, 389, 390, 391, 392, 393, 395, 396, 397, 398, 399, 400, 401, 402, 404, 406, 408, 410, 411, 412, 413, 414, 415, 416, 419, 420, 421, 422, 423, 424, 425, 426, 428, 429, 430, 432, 433, 434, 435, 438, 439, 440, 441, 443, 444, 445, 446, 447, 448, 449, 450, 454, 455, 456, 457, 458, 461, 463, 464, 465, 466, 467, 468, 471, 476, 478, 479, 480, 481, 485, 486, 488, 490, 491, 492, 494, 495, 496, 497, 499, 500, 501, 502, 503, 504, 506, 507, 508, 510, 511, 512, 513, 514, 515, 516, 517, 518, 520, 521, 524, 525, 526, 527, 528, 529, 530, 531, 532, 534, 535, 536, 537, 538, 540, 541, 542, 543, 544, 546, 547, 548, 549, 550, 551, 552, 554, 555, 556, 557, 558, 560, 561, 562, 563, 565, 566, 567, 568, 569, 570, 571, 572, 573, 574, 575, 576, 577, 578, 579, 580, 581, 582, 583, 585, 587, 588, 589, 590, 591, 592, 593, 595, 596, 597, 598, 599, 600, 601, 602, 605, 606, 609, 610, 611, 612, 613, 615, 616, 617, 619, 621, 623, 624, 625, 628, 629, 630, 631, 633, 634, 635, 636, 637, 638, 639, 640, 641, 643, 644, 645, 647, 648, 649, 650, 651, 652, 653, 654, 657, 658, 659, 660, 661, 664, 665, 666, 668, 669, 670, 671, 672, 674, 675, 676, 677, 679, 681, 682, 685, 686, 687, 688, 689, 690, 691, 692, 694, 695, 697, 698, 699, 700, 701, 703, 705, 709, 711, 712, 713, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 728, 731, 732, 733, 734, 736, 737, 738, 739, 741, 742, 744, 745, 746, 747, 748, 749, 751, 752, 753, 754, 756, 757, 759, 761, 762, 763, 764, 765, 767, 769, 770, 773, 775, 776, 777, 779, 781, 782, 783, 784, 785, 786, 787, 788, 789, 790, 791, 792, 793, 794, 795, 796, 797, 798, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 812, 813, 815, 817, 818, 819, 820, 821, 822, 823, 825, 826, 827, 828, 829, 831, 832, 833, 834, 835, 836, 837, 838, 841, 842, 844, 845, 846, 849, 850, 851, 853, 857, 859, 861, 863, 864, 865, 866, 867, 868, 869, 873, 874, 875, 876, 877, 882, 883, 884, 885, 887, 888, 889, 891, 892, 894, 896, 897, 898, 899, 900, 903, 905, 908, 909, 910, 911, 912, 913, 916, 917, 918, 919, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 931, 932, 933, 934, 936, 937, 939, 940, 941, 942, 944, 945, 946, 947, 950, 952, 954, 955, 957, 958, 959, 961, 962, 963, 964, 965, 967, 968, 971, 972, 973, 975, 976, 977, 980, 981, 982, 983, 984, 986, 987, 988, 989, 990, 991, 995, 997, 999, 1001, 1002, 1003, 1004, 1005, 1006, 1009, 1010, 1012, 1013, 1014, 1015, 1016, 1020, 1024, 1026, 1027, 1028, 1031, 1032, 1034, 1035, 1036, 1037, 1040, 1041, 1042, 1044, 1045, 1046, 1047, 1048, 1049, 1050, 1051, 1052, 1053, 1054, 1056, 1057, 1060, 1063, 1065, 1066, 1067, 1068, 1069, 1070, 1071, 1072, 1073, 1074, 1075, 1076, 1077, 1078, 1081, 1082, 1083, 1085, 1086, 1087, 1088, 1089, 1091, 1092, 1093, 1094, 1095, 1096, 1097, 1098, 1100, 1101, 1102, 1103, 1105, 1106, 1107, 1108, 1109, 1110, 1111, 1113, 1114, 1116, 1117, 1119, 1121, 1122, 1123, 1124, 1126, 1128, 1129, 1130, 1131, 1132, 1134, 1136, 1138, 1139, 1140, 1141, 1142, 1143, 1144, 1145, 1146, 1147, 1149, 1150, 1151, 1152, 1153, 1154, 1155, 1156, 1158, 1159, 1160, 1161, 1162, 1163, 1164, 1165, 1167, 1168, 1169, 1170, 1171, 1172, 1175, 1176, 1177, 1178, 1179, 1181, 1182, 1183, 1184, 1186, 1188, 1189, 1190, 1191, 1192, 1193, 1194, 1195, 1196, 1197, 1200, 1201, 1202, 1204, 1207, 1209, 1212, 1213, 1214, 1215]
    cafca_subject_ids_val: [1216, 1217, 1218, 1220, 1221, 1222, 1224, 1226, 1227, 1230, 1232, 1233, 1234, 1236, 1237, 1240, 1241, 1242, 1243, 1245, 1247, 1250, 1251, 1253, 1255, 1256, 1257, 1258, 1259, 1261, 1262, 1263, 1264, 1266, 1267, 1268, 1269, 1270, 1271, 1272, 1273, 1274, 1275, 1276, 1277, 1278, 1279, 1280, 1282, 1283, 1284, 1286, 1287, 1289, 1290, 1291, 1292, 1293, 1295, 1296, 1297, 1298, 1303, 1304, 1306, 1307, 1308, 1309, 1311, 1314, 1315, 1316, 1318, 1320, 1322, 1325, 1326, 1328, 1329, 1330, 1331, 1333, 1334, 1335, 1337, 1338, 1339, 1341, 1343, 1344, 1345, 1347, 1349, 1350, 1351, 1353, 1354, 1356, 1357, 1359]
  