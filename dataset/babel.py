#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# Attempt to import rotation conversion utilities from PyTorch3D
try:
    from pytorch3d.transforms import axis_angle_to_matrix as aa_to_rotmat_pytorch3d
    from pytorch3d.transforms import matrix_to_rotation_6d as rotmat_to_6d_pytorch3d
    print("Using PyTorch3D for axis-angle to 6D rotation conversion.") #
except ImportError:
    print("Critical Error: PyTorch3D not found. This script requires PyTorch3D for rotation conversions.")
    print("Please install PyTorch3D: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")
    exit()

# Helper function to convert axis-angle to 6D (used for root_orient in train/val/test for seq_6d)
def convert_single_joint_aa_to_6d(aa_tensor):
    """Converts a single joint's axis-angle (T, 3) to 6D (T, 1, 6)."""
    T = aa_tensor.shape[0]
    if T == 0: # Handle empty tensor case
        return torch.empty(0, 1, 6, dtype=aa_tensor.dtype, device=aa_tensor.device)
    rot_mats = aa_to_rotmat_pytorch3d(aa_tensor) # (T, 3, 3)
    rot_mats_reshaped = rot_mats.unsqueeze(1) # (T, 1, 3, 3)
    return rotmat_to_6d_pytorch3d(rot_mats_reshaped) # (T, 1, 6)


class BabelTotalCaptureDataset(Dataset):
    def __init__(self,
                 data_dir,
                 split='train',
                 max_seq_len=64,
                 num_betas=16,
                 num_body_joints=21):
        self.data_dir = data_dir #
        self.split = split #
        self.max_seq_len = max_seq_len #
        self.num_betas = num_betas #
        self.num_body_joints = num_body_joints #
        self.num_root_joints = 1 # For the root orientation #

        sample_rate_in_filename = 1 # Match with preprocess_babel.py #
        self.data_file = os.path.join(
            data_dir,
            f"babel_totalcapture_{split}_rootAA_body6D_trans_betas" # This was from your preprocess script
            f"_sr{sample_rate_in_filename}_betas{self.num_betas}.npz"
        ) #

        self.processed_pose_body_6d = [] #
        self.processed_root_orient_6d = [] # Will always store 6D root for seq_6d consistency [MODIFIED]

        # For test split, additionally store original root_orient_aa and other SMPL params
        self.processed_root_orient_aa_orig_test = [] # [MODIFIED]
        self.processed_trans_test = [] # [MODIFIED]
        self.processed_betas_test = [] # [MODIFIED]
        self.processed_gender_test = [] # [MODIFIED]
        self.processed_babel_sids_test = [] # [MODIFIED]
        self.processed_frame_counts_test = [] # [MODIFIED]


        if not os.path.exists(self.data_file): #
            print(f"Error: Processed data file not found at {self.data_file}") #
            return

        print(f"Loading data from: {self.data_file}") #
        data_npz = np.load(self.data_file, allow_pickle=True) #

        all_pose_body_6d_from_file = data_npz['pose_body_6d'] #
        all_root_orient_aa_from_file = data_npz['root_orient_aa'] #

        # Load these only if they exist, especially for test split data
        all_trans_from_file = data_npz.get('trans') #
        all_betas_from_file = data_npz.get('betas') #
        all_genders_from_file = data_npz.get('gender') #
        all_babel_sids_from_file = data_npz.get('babel_sid') #
        all_frame_counts_from_file = data_npz.get('frame_counts') #

        for i in tqdm(range(len(all_pose_body_6d_from_file)), desc=f"Processing sequences for {split} split"): #
            seq_pose_body_6d_np = all_pose_body_6d_from_file[i] #
            seq_root_orient_aa_np = all_root_orient_aa_from_file[i] # This is (T, 3) #

            if seq_root_orient_aa_np.shape[0] == 0 or seq_pose_body_6d_np.shape[0] == 0:
                print(f"Warning: Empty sequence found at index {i} for {self.split} split. Skipping.")
                continue


            # Always convert root_orient_aa to 6D for the main seq_6d
            seq_root_orient_6d_torch = convert_single_joint_aa_to_6d(torch.from_numpy(seq_root_orient_aa_np).float()) #

            # Split sequences into fixed-length clips
            split_seqs_pose_body_6d = self._split_seq(seq_pose_body_6d_np, self.max_seq_len, K=self.num_body_joints, D=6) #
            split_seqs_root_orient_6d = self._split_seq(seq_root_orient_6d_torch.numpy(), self.max_seq_len, K=self.num_root_joints, D=6) #

            self.processed_pose_body_6d.extend(split_seqs_pose_body_6d) #
            self.processed_root_orient_6d.extend(split_seqs_root_orient_6d) # [MODIFIED]

            num_clips_generated = len(split_seqs_pose_body_6d) #
            if num_clips_generated == 0: # If original sequence was shorter than any possible clip after split
                continue


            if self.split == 'test': #
                # Store original root_orient_aa for the dictionary
                split_seqs_root_orient_aa_orig = self._split_seq(seq_root_orient_aa_np, self.max_seq_len, K=self.num_root_joints, D=3) # [MODIFIED]
                self.processed_root_orient_aa_orig_test.extend(split_seqs_root_orient_aa_orig) # [MODIFIED]

                seq_trans_np = all_trans_from_file[i] if all_trans_from_file is not None and i < len(all_trans_from_file) else None # [MODIFIED]
                seq_betas_np = all_betas_from_file[i] if all_betas_from_file is not None and i < len(all_betas_from_file) else None # [MODIFIED]
                gender_str = all_genders_from_file[i] if all_genders_from_file is not None and i < len(all_genders_from_file) else "neutral" # [MODIFIED]
                babel_sid_str = all_babel_sids_from_file[i] if all_babel_sids_from_file is not None and i < len(all_babel_sids_from_file) else "unknown_sid" # [MODIFIED]
                frame_count_int = all_frame_counts_from_file[i] if all_frame_counts_from_file is not None and i < len(all_frame_counts_from_file) else seq_root_orient_aa_np.shape[0] # [MODIFIED]


                if seq_trans_np is not None: #
                    split_seqs_trans = self._split_seq(seq_trans_np, self.max_seq_len, K=self.num_root_joints, D=3) #
                    self.processed_trans_test.extend(split_seqs_trans) # [MODIFIED]
                else:
                    self.processed_trans_test.extend([np.zeros((self.max_seq_len, self.num_root_joints, 3))] * num_clips_generated) # [MODIFIED]

                if seq_betas_np is not None: #
                    self.processed_betas_test.extend([seq_betas_np] * num_clips_generated) # [MODIFIED]
                else:
                    self.processed_betas_test.extend([np.zeros(self.num_betas)] * num_clips_generated) # [MODIFIED]

                self.processed_gender_test.extend([gender_str] * num_clips_generated) # [MODIFIED]
                self.processed_babel_sids_test.extend([babel_sid_str] * num_clips_generated) # [MODIFIED]
                self.processed_frame_counts_test.extend([frame_count_int] * num_clips_generated) # [MODIFIED]

        # Sanity checks
        if self.split == 'test': #
            if not (len(self.processed_pose_body_6d) == len(self.processed_root_orient_6d) == \
                    len(self.processed_root_orient_aa_orig_test) == \
                    len(self.processed_trans_test) == len(self.processed_betas_test) == \
                    len(self.processed_gender_test) == len(self.processed_babel_sids_test) == \
                    len(self.processed_frame_counts_test)): # [MODIFIED]
                print("Warning: Mismatch in processed list lengths for test split! Check for None values in source NPZ or empty sequences.") # [MODIFIED]
                print(f"Lengths: pose_body={len(self.processed_pose_body_6d)}, root_6d={len(self.processed_root_orient_6d)}, "
                      f"root_aa_orig={len(self.processed_root_orient_aa_orig_test)}, trans={len(self.processed_trans_test)}, "
                      f"betas={len(self.processed_betas_test)}") # [MODIFIED]

        else: # train/val #
            assert len(self.processed_pose_body_6d) == len(self.processed_root_orient_6d), \
                   "Mismatch in processed pose_body_6d and root_orient_6d list lengths!" # [MODIFIED]


    def __len__(self): #
        return len(self.processed_pose_body_6d) #

    def __getitem__(self, idx): #
        pose_body_6d_clip = torch.from_numpy(self.processed_pose_body_6d[idx]).float() # (max_len, num_body_joints, 6) #
        root_orient_6d_clip = torch.from_numpy(self.processed_root_orient_6d[idx]).float() # (max_len, 1, 6) # [MODIFIED]

        # Concatenate root_orient_6d and pose_body_6d to form seq_6d for all splits
        seq_6d = torch.cat((root_orient_6d_clip, pose_body_6d_clip), dim=1) # [MODIFIED]


        if self.split in ['train', 'val']: #
            # Placeholder for seq_xyz and label as these are not currently preprocessed/loaded
            seq_xyz = torch.zeros_like(seq_6d[..., :3]) # Placeholder (max_len, num_total_joints, 3) #
            label = 0 # Placeholder label #
            return seq_6d, seq_xyz, label #

        else: # self.split == 'test' #
            item = { #
                'seq_6d': seq_6d, # [MODIFIED] - Main input for model consistency
                'pose_body_6d': pose_body_6d_clip, # [MODIFIED] - Retained for separate access if needed
                'root_orient_aa_orig': torch.from_numpy(self.processed_root_orient_aa_orig_test[idx]).float(), # (max_len, 1, 3) [MODIFIED]
            }

            current_trans = self.processed_trans_test[idx] # [MODIFIED]
            if current_trans is not None: #
                item['trans'] = torch.from_numpy(current_trans).float() #
            else: #
                item['trans'] = torch.zeros(self.max_seq_len, self.num_root_joints, 3) # Placeholder if None #

            current_betas = self.processed_betas_test[idx] # [MODIFIED]
            if current_betas is not None: #
                 item['betas'] = torch.from_numpy(np.asarray(current_betas, dtype=np.float32)).float() #
            else: #
                item['betas'] = torch.zeros(self.num_betas) # Placeholder if None #

            item['gender'] = self.processed_gender_test[idx] # [MODIFIED]
            item['babel_sid'] = self.processed_babel_sids_test[idx] # [MODIFIED]
            item['frame_count_orig_sampled'] = self.processed_frame_counts_test[idx] # [MODIFIED]

            return item #

    def _split_seq(self, seq, max_len, K=1, D=3): # K=num_joints, D=dim_per_joint #
        if seq is None: #
            return []

        T_orig = seq.shape[0] #
        if T_orig == 0: # Handle case where sequence is empty from the start
            return []

        # Ensure seq is (T, K, D)
        if len(seq.shape) == 2: # Input is (T, D_flat) #
            if seq.shape[1] == K*D: # e.g. (T, 21*6) or (T, 1*3) #
                seq = seq.reshape(T_orig, K, D) #
            elif K==1 and seq.shape[1] == D : # e.g. (T,3) for K=1, D=3 #
                seq = seq.reshape(T_orig, K, D) #
            else: #
                raise ValueError(f"Cannot reshape sequence of shape {seq.shape} to (T, K={K}, D={D})")
        elif len(seq.shape) == 3: # Input is already (T,K,D) #
            assert seq.shape[1] == K and seq.shape[2] == D, \
                f"Input seq shape {seq.shape} doesn't match K={K}, D={D}" #
        else: #
            raise ValueError(f"Unexpected sequence shape: {seq.shape}")


        num_clips_total = (T_orig + max_len - 1) // max_len # Ceiling division to include all frames #
        clipped_seqs = [] #

        for i in range(num_clips_total): #
            start_idx = i * max_len #
            end_idx = start_idx + max_len #

            clip_orig = seq[start_idx:end_idx] #
            current_clip_len = clip_orig.shape[0] #

            if current_clip_len < max_len: #
                padding_needed = max_len - current_clip_len #
                # Pad with the last frame's data (replication padding)
                padding_frames = np.repeat(clip_orig[-1:, :, :], padding_needed, axis=0) #
                clip_padded = np.concatenate([clip_orig, padding_frames], axis=0) #
                clipped_seqs.append(clip_padded) #
            else: #
                clipped_seqs.append(clip_orig) #

        if not clipped_seqs and T_orig > 0 : # Should not happen with ceiling division logic #
             pass #


        return clipped_seqs #


if __name__ == "__main__": #
    print(f"Current working directory: {os.getcwd()}") #
    example_data_dir = "./data/babel_processed/totalcapture_body_aa_root_splits/" # ADJUST AS NEEDED #

    print(f"Attempting to load 'train' split from: {example_data_dir}") #
    try: #
        train_dataset = BabelTotalCaptureDataset(data_dir=example_data_dir, split='train', max_seq_len=60, num_betas=16) #
        if len(train_dataset) > 0: #
            print(f"Train dataset length: {len(train_dataset)}") #
            seq_6d, seq_xyz, label = train_dataset[0] # #
            print("\nTrain item example (mimicking Pw3dVQVAESixDDataSet):") #
            print(f"  seq_6d shape: {seq_6d.shape}") # Expected: (max_len, 1+num_body_joints, 6) #
            print(f"  seq_xyz shape (placeholder): {seq_xyz.shape}") # #
            print(f"  label (placeholder): {label}") # #
        else: #
            print("Train dataset is empty or failed to load. Check paths and preprocessing.") #
    except Exception as e: #
        print(f"Error loading/processing train dataset: {e}") #


    print(f"\nAttempting to load 'test' split from: {example_data_dir}") #
    try: #
        test_dataset = BabelTotalCaptureDataset(data_dir=example_data_dir, split='test', max_seq_len=60, num_betas=16) #
        if len(test_dataset) > 0: #
            print(f"Test dataset length: {len(test_dataset)}") #
            test_item = test_dataset[0] # #
            print("\nTest item example (dictionary):") #
            for key, value in test_item.items(): #
                if isinstance(value, torch.Tensor): #
                    print(f"  {key} shape: {value.shape}") #
                else: #
                    print(f"  {key}: {value}") #
        else: #
            print("Test dataset is empty or failed to load. Check paths and preprocessing.") #
    except Exception as e: #
        print(f"Error loading/processing test dataset: {e}") #