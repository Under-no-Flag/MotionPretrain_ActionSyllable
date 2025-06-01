import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# 确保 PyTorch3D 可用，因为我们需要将 global_orient_aa 转换为 6D
try:
    from pytorch3d.transforms import axis_angle_to_matrix as aa_to_rotmat_pytorch3d
    from pytorch3d.transforms import matrix_to_rotation_6d as rotmat_to_6d_pytorch3d
    # print("Using PyTorch3D for axis-angle to 6D rotation conversion in Pw3DDataset.")
except ImportError:
    print("Critical Error: PyTorch3D not found. This script requires PyTorch3D for rotation conversions.")
    print("Please install PyTorch3D: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")
    exit()


def convert_single_joint_aa_to_6d(aa_tensor):
    """Converts a single joint's axis-angle (T, 3) to 6D (T, 1, 6)."""
    T = aa_tensor.shape[0]
    if T == 0:
        return torch.empty(0, 1, 6, dtype=aa_tensor.dtype, device=aa_tensor.device)
    rot_mats = aa_to_rotmat_pytorch3d(aa_tensor)  # (T, 3, 3)
    rot_mats_reshaped = rot_mats.unsqueeze(1)  # (T, 1, 3, 3)
    return rotmat_to_6d_pytorch3d(rot_mats_reshaped)  # (T, 1, 6)


class Pw3dSmplParamDataset(Dataset):  # 新的类名或重命名 Pw3dVQVAESixDDataSet
    def __init__(self,
                 data_dir,
                 split='train',
                 max_seq_len=64,
                 sample_rate=2,  # 需要匹配预处理时的采样率以正确加载文件名
                 num_betas=10,  # 3DPW通常是10个betas
                 num_body_joints=23,  # SMPL body joints
                 num_total_joints=24):  # 1 (root) + 23 (body)
        """
        Dataset class for preprocessed 3DPW data (SMPL parameters).

        Args:
            data_dir (str): Path to the directory containing the processed .npz files.
            split (str): 'train', 'validation', or 'test'. (3DPW uses 'validation')
            max_seq_len (int): Maximum sequence length for fixed-size batches.
            sample_rate (int): Sample rate used during preprocessing (for filename).
            num_betas (int): Number of beta parameters.
            num_body_joints (int): Number of SMPL body joints (excluding root).
            num_total_joints (int): Total number of joints in the concatenated 6D pose (root + body).
        """
        self.data_dir = data_dir
        self.split = split
        if self.split == "val":  # 3DPW conventionally uses "validation"
            self.split = "validation"
        self.max_seq_len = max_seq_len
        self.num_betas = num_betas
        self.num_body_joints = num_body_joints
        self.num_total_joints = num_total_joints  # Should be 1 (root) + num_body_joints
        self.num_root_joints = 1

        self.data_file = os.path.join(
            data_dir,
            f"3dpw_{self.split}_rootAA_body6D_transl_betas"
            f"_sr{sample_rate}_betas{self.num_betas}.npz"
        )

        self.processed_seq_6d_concatenated = []  # Stores concatenated root_6d + body_6d

        # For test split, store original global_orient_aa and other SMPL params
        self.processed_global_orient_aa_orig_test = []
        self.processed_body_pose_6d_test = []  # Body pose 6D separately for test dict
        self.processed_transl_test = []
        self.processed_betas_test = []
        self.processed_gender_test = []
        self.processed_seq_name_test = []
        self.processed_frame_counts_test = []

        # Placeholders for items needed by Pw3dVQVAESixDDataSet format
        # self.processed_xyz_placeholder = [] # TODO: Implement if needed
        # self.processed_label_placeholder = [] # TODO: Implement if needed

        if not os.path.exists(self.data_file):
            print(f"Error: Processed data file not found at {self.data_file}")
            print("Please ensure you have run the modified preprocess_3dpw.py and the path/parameters are correct.")
            return

        print(f"Loading data from: {self.data_file}")
        data_npz = np.load(self.data_file, allow_pickle=True)

        all_global_orient_aa_from_file = data_npz['global_orient_aa']  # (N_seqs, T, 3)
        all_body_pose_6d_from_file = data_npz['body_pose_6d']  # (N_seqs, T, num_body_joints, 6)

        all_transl_from_file = data_npz.get('transl')
        all_betas_from_file = data_npz.get('betas')  # (N_seqs, num_betas)
        all_genders_from_file = data_npz.get('gender')
        all_seq_names_from_file = data_npz.get('seq_name')
        all_frame_counts_from_file = data_npz.get('frame_counts')

        for i in tqdm(range(len(all_global_orient_aa_from_file)), desc=f"Processing sequences for {self.split} split"):
            seq_global_orient_aa_np = all_global_orient_aa_from_file[i]  # (T, 3)
            seq_body_pose_6d_np = all_body_pose_6d_from_file[i]  # (T, num_body_joints, 6)

            if seq_global_orient_aa_np.shape[0] == 0 or seq_body_pose_6d_np.shape[0] == 0:
                print(f"Warning: Empty sequence found at index {i} for {self.split} split. Skipping.")
                continue

            # Convert global_orient_aa to 6D
            seq_global_orient_6d_torch = convert_single_joint_aa_to_6d(
                torch.from_numpy(seq_global_orient_aa_np).float())
            seq_global_orient_6d_np = seq_global_orient_6d_torch.numpy()  # (T, 1, 6)

            # Concatenate root_6d and body_6d (numpy arrays)
            # Ensure body_pose_6d_np is (T, K, D) before cat
            if len(seq_body_pose_6d_np.shape) == 2:  # Should not happen if preprocess saved (T,K,D)
                seq_body_pose_6d_np = seq_body_pose_6d_np.reshape(seq_body_pose_6d_np.shape[0], self.num_body_joints, 6)

            seq_6d_concatenated_np = np.concatenate((seq_global_orient_6d_np, seq_body_pose_6d_np),
                                                    axis=1)  # (T, 1+num_body_joints, 6)

            # Split into clips
            split_seqs_6d_concatenated = self._split_seq(seq_6d_concatenated_np, self.max_seq_len,
                                                         K=self.num_total_joints, D=6)
            self.processed_seq_6d_concatenated.extend(split_seqs_6d_concatenated)

            num_clips_generated = len(split_seqs_6d_concatenated)
            if num_clips_generated == 0:
                continue

            # Placeholders for labels and xyz
            # seq_label_dummy = all_seq_names_from_file[i] if all_seq_names_from_file is not None else "unknown"
            # self.processed_label_placeholder.extend([seq_label_dummy] * num_clips_generated) # Using seq_name as a dummy label
            # self.processed_xyz_placeholder.extend([np.zeros((self.max_seq_len, self.num_total_joints, 3))] * num_clips_generated)

            if self.split == 'test':
                # Store original global_orient_aa for the dictionary
                split_seqs_global_orient_aa_orig = self._split_seq(seq_global_orient_aa_np, self.max_seq_len,
                                                                   K=self.num_root_joints, D=3)
                self.processed_global_orient_aa_orig_test.extend(split_seqs_global_orient_aa_orig)

                # Store body_pose_6d separately for test dictionary if needed
                split_seqs_body_pose_6d = self._split_seq(seq_body_pose_6d_np, self.max_seq_len, K=self.num_body_joints,
                                                          D=6)
                self.processed_body_pose_6d_test.extend(split_seqs_body_pose_6d)

                seq_transl_np = all_transl_from_file[i] if all_transl_from_file is not None and i < len(
                    all_transl_from_file) else None
                seq_betas_np = all_betas_from_file[i] if all_betas_from_file is not None and i < len(
                    all_betas_from_file) else None
                gender_str = all_genders_from_file[i] if all_genders_from_file is not None and i < len(
                    all_genders_from_file) else "neutral"
                seq_name_str = all_seq_names_from_file[i] if all_seq_names_from_file is not None and i < len(
                    all_seq_names_from_file) else "unknown_sid"
                frame_count_int = all_frame_counts_from_file[i] if all_frame_counts_from_file is not None and i < len(
                    all_frame_counts_from_file) else seq_global_orient_aa_np.shape[0]

                if seq_transl_np is not None:
                    split_seqs_transl = self._split_seq(seq_transl_np, self.max_seq_len, K=self.num_root_joints, D=3)
                    self.processed_transl_test.extend(split_seqs_transl)
                else:
                    self.processed_transl_test.extend(
                        [np.zeros((self.max_seq_len, self.num_root_joints, 3))] * num_clips_generated)

                if seq_betas_np is not None:
                    self.processed_betas_test.extend([seq_betas_np] * num_clips_generated)
                else:
                    self.processed_betas_test.extend([np.zeros(self.num_betas)] * num_clips_generated)

                self.processed_gender_test.extend([gender_str] * num_clips_generated)
                self.processed_seq_name_test.extend([seq_name_str] * num_clips_generated)
                self.processed_frame_counts_test.extend([frame_count_int] * num_clips_generated)

        # Sanity checks
        if self.split == 'test':
            if not (len(self.processed_seq_6d_concatenated) == \
                    len(self.processed_global_orient_aa_orig_test) == \
                    len(self.processed_body_pose_6d_test) == \
                    len(self.processed_transl_test) == len(self.processed_betas_test) == \
                    len(self.processed_gender_test) == len(self.processed_seq_name_test) == \
                    len(self.processed_frame_counts_test)):
                print("Warning: Mismatch in processed list lengths for 3DPW test split!")
                # Detailed print for debugging
                print(f"  seq_6d: {len(self.processed_seq_6d_concatenated)}, "
                      f"g_orient_aa: {len(self.processed_global_orient_aa_orig_test)}, "
                      f"body_6d: {len(self.processed_body_pose_6d_test)}, "
                      f"transl: {len(self.processed_transl_test)}, "
                      f"betas: {len(self.processed_betas_test)}")
        # else: # train/val
        # Add relevant assertions for train/val if placeholders are used and populated

    def __len__(self):
        return len(self.processed_seq_6d_concatenated)

    def __getitem__(self, idx):
        seq_6d_clip = torch.from_numpy(
            self.processed_seq_6d_concatenated[idx]).float()  # (max_len, num_total_joints, 6)

        if self.split in ['train', 'validation']:  # 3DPW uses 'validation'
            # Mimic Pw3dVQVAESixDDataSet return format for train/val

            # TODO: Implement loading or generation of seq_xyz (ground truth 3D coordinates)
            # For now, returning a placeholder. This needs to be addressed for MPJPE calculation.
            # One option is to modify preprocess_3dpw.py to also save relative XYZ coordinates.
            seq_xyz_placeholder = torch.zeros_like(seq_6d_clip[..., :3])  # (max_len, num_total_joints, 3)

            # TODO: Implement loading of actual labels if available and needed
            # For now, using a dummy integer label. Original Pw3dVQVAESixDDataSet used seq_name,
            # but your target returns an integer. If seq_name is needed, load from self.processed_seq_name_test[idx]
            # (though this list is only populated for 'test' split currently).
            label_placeholder = 0  # Dummy integer label

            return seq_6d_clip, seq_xyz_placeholder, label_placeholder

        else:  # self.split == 'test'
            item = {
                'seq_6d': seq_6d_clip,  # Consistent 6D input (root_6d + body_6d)
                'root_orient_aa_orig': torch.from_numpy(self.processed_global_orient_aa_orig_test[idx]).float(),
                'body_pose_6d': torch.from_numpy(self.processed_body_pose_6d_test[idx]).float(),
                # Original body_pose_6d part
                'trans': torch.from_numpy(self.processed_transl_test[idx]).float().squeeze(1),
                'betas': torch.from_numpy(np.asarray(self.processed_betas_test[idx], dtype=np.float32)).float(),
                'gender': self.processed_gender_test[idx],
                'seq_name': self.processed_seq_name_test[idx],  # Previously label_seq
                'frame_count_orig_sampled': self.processed_frame_counts_test[idx]
            }
            return item

    def _split_seq(self, seq, max_len, K=1, D=3):
        if seq is None or seq.shape[0] == 0:
            return []

        T_orig = seq.shape[0]

        if len(seq.shape) == 2:
            if seq.shape[1] == K * D:
                seq = seq.reshape(T_orig, K, D)
            elif K == 1 and seq.shape[1] == D:
                seq = seq.reshape(T_orig, K, D)
            else:
                raise ValueError(f"Cannot reshape sequence of shape {seq.shape} to (T, K={K}, D={D})")
        elif len(seq.shape) == 3:
            assert seq.shape[1] == K and seq.shape[2] == D, f"Input seq shape {seq.shape} doesn't match K={K}, D={D}"
        else:
            raise ValueError(f"Unexpected sequence shape: {seq.shape}")

        num_clips_total = (T_orig + max_len - 1) // max_len
        clipped_seqs = []

        for i in range(num_clips_total):
            start_idx = i * max_len
            end_idx = start_idx + max_len
            clip_orig = seq[start_idx:end_idx]
            current_clip_len = clip_orig.shape[0]

            if current_clip_len < max_len:
                padding_needed = max_len - current_clip_len
                padding_frames = np.repeat(clip_orig[-1:, :, :], padding_needed, axis=0)
                clip_padded = np.concatenate([clip_orig, padding_frames], axis=0)
                clipped_seqs.append(clip_padded)
            else:
                clipped_seqs.append(clip_orig)

        return clipped_seqs


if __name__ == "__main__":
    print(f"Current working directory: {os.getcwd()}")
    # Adjust data_dir to where your **newly preprocessed** 3DPW files are
    # (the ones with rootAA, body6D, transl, betas)
    example_data_dir = "../data/3dpw/processed_params/"  # ADJUST AS NEEDED
    sample_rate_for_test = 2  # Must match the sr used in the filename
    num_betas_for_test = 10  # Must match the betas count used in the filename

    print(f"Attempting to load 'train' split from: {example_data_dir}")
    try:
        # Note: Pw3dVQVAESixDDataSet was the old name, using Pw3dSmplParamDataset now
        train_dataset = Pw3dSmplParamDataset(
            data_dir=example_data_dir,
            split='train',
            max_seq_len=60,
            sample_rate=sample_rate_for_test,
            num_betas=num_betas_for_test
        )
        if len(train_dataset) > 0:
            print(f"Train dataset length: {len(train_dataset)}")
            seq_6d, seq_xyz, label = train_dataset[0]
            print("\nTrain item example (mimicking Pw3dVQVAESixDDataSet output style):")
            print(f"  seq_6d shape: {seq_6d.shape}")  # Expected: (max_len, 24, 6)
            print(f"  seq_xyz shape (placeholder): {seq_xyz.shape}")
            print(f"  label (placeholder): {label}")
        else:
            print("Train dataset is empty. Check paths and preprocessing (run modified preprocess_3dpw.py).")
    except Exception as e:
        print(f"Error loading/processing train dataset: {e}")
        import traceback

        traceback.print_exc()

    print(f"\nAttempting to load 'test' split from: {example_data_dir}")
    try:
        test_dataset = Pw3dSmplParamDataset(
            data_dir=example_data_dir,
            split='test',
            max_seq_len=60,
            sample_rate=sample_rate_for_test,
            num_betas=num_betas_for_test
        )
        if len(test_dataset) > 0:
            print(f"Test dataset length: {len(test_dataset)}")
            test_item = test_dataset[0]
            print("\nTest item example (dictionary):")
            for key, value in test_item.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key} shape: {value.shape}")
                else:
                    print(f"  {key}: {value}")
        else:
            print("Test dataset is empty. Check paths and preprocessing (run modified preprocess_3dpw.py).")
    except Exception as e:
        print(f"Error loading/processing test dataset: {e}")
        import traceback

        traceback.print_exc()