# dataset/cafca_lam_dataset.py

import os
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset


from lam.dataset import env_paths

class CafcaLamDataset(Dataset):
    def __init__(self, subject_list, mode="lam_infer", preprocessed_subdir="preprocessed_1024"):
        """
        Dataset for loading preprocessed CAFCA data for LAM inference.

        Args:
            subject_list (list): List of subject IDs (integers) to load.
            mode (str): Mode of operation (e.g., "lam_infer"). Not strictly used for data loading logic here.
            preprocessed_subdir (str): Name of the subdirectory containing the 1024x1024 preprocessed data.
        """
        self.root_dir = Path(env_paths.DATA_DIR)
        self.subject_list = subject_list
        self.preprocessed_subdir_name = preprocessed_subdir
        self.data = []

        for subject_int in subject_list:
            subject_str_zfill = str(subject_int).zfill(5)
            subject_base_dir = (
                self.root_dir
                / subject_str_zfill
                / f"{env_paths.EXPRESSION_ID}_{env_paths.ENVIRONMENT_ID}"
            )

            preprocessed_data_path = subject_base_dir / self.preprocessed_subdir_name
            flame_params_path = subject_base_dir / f"{subject_str_zfill}{env_paths.FLAME_FILENAME}"
            
            # Paths to the preprocessed data components
            processed_images_dir = preprocessed_data_path / "images"
            processed_masks_dir = preprocessed_data_path / "masks"
            processed_cameras_dir = preprocessed_data_path / "cameras_json"

            if not flame_params_path.exists():
                print(f"Warning: Canonical FLAME param file not found for subject {subject_str_zfill} at {flame_params_path}. Skipping subject.")
                continue

            if not processed_images_dir.exists():
                print(f"Warning: Preprocessed images directory not found for subject {subject_str_zfill} at {processed_images_dir}. Skipping subject.")
                continue

            # Iterate through camera files in the *preprocessed* camera directory
            # to ensure we only load data for which preprocessing was successful.
            processed_camera_files = sorted(list(processed_cameras_dir.glob("*.json")))

            if not processed_camera_files:
                print(f"Warning: No preprocessed camera files found for subject {subject_str_zfill} in {processed_cameras_dir}. Skipping subject.")
                continue

            for cam_json_file in processed_camera_files:
                cam_id = cam_json_file.stem # e.g., C02

                image_file = processed_images_dir / f"{cam_id}.png"
                mask_file = processed_masks_dir / f"{cam_id}.png"

                if not image_file.exists():
                    print(f"Warning: Preprocessed image {image_file} not found for subject {subject_str_zfill}, cam {cam_id}. Skipping item.")
                    continue
                if not mask_file.exists():
                    print(f"Warning: Preprocessed mask {mask_file} not found for subject {subject_str_zfill}, cam {cam_id}. Skipping item.")
                    continue

                try:
                    with open(cam_json_file, "r") as f:
                        cam_params = json.load(f)

                    if "K" not in cam_params:
                        raise KeyError(f"'K' key not found in preprocessed camera file {cam_json_file}")

                    self.data.append(
                        {
                            "subject_id_int": subject_int,
                            "subject_id_str": subject_str_zfill,
                            "cam_id": cam_id,
                            "image_file_path": str(image_file),  # Path to 1024x1024 preprocessed image
                            "mask_file_path": str(mask_file),    # Path to 1024x1024 preprocessed mask
                            "intrinsic": np.array(cam_params["K"]), # Adjusted intrinsics for 1024x1024 image
                            "canonical_flame_param_path": str(flame_params_path),
                            "cam_2_world": np.array(cam_params["cam2world"]), # Camera to canonical flame transformation
                        }
                    )
                except Exception as e:
                    print(f"Error loading data for subject {subject_str_zfill}, cam {cam_id}: {e}")
        
        if not self.data:
            print(f"Warning: {self.__class__.__name__} initialized with no data items. Check paths and preprocessed_subdir: '{self.preprocessed_subdir_name}'.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a dictionary containing paths and raw data for a single item.
        The LAM inference pipeline will load images/masks from these paths.
        """
        item = self.data[idx]
        return item

if __name__ == '__main__':
    if hasattr(env_paths, 'subjects_train') and env_paths.subjects_train:
        dataset = CafcaLamDataset(subject_list=env_paths.subjects_train)
        print(f"Loaded {len(dataset)} items for LAM inference.")
        if len(dataset) > 0:
            print("First item:", dataset[0])
    else:
        print("Please define 'subjects_train' in your dataset/env_paths.py (e.g., subjects_train = [30])")