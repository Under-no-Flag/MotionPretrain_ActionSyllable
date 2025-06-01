#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-process BABEL dataset to extract TotalCapture sequences.
Converts pose_body to 6D rotation representation.
Retains root_orient in axis-angle format, trans, and betas.
Excludes hand poses.
Saves separate files for train, val, and test splits.
"""
import argparse
import json
import pathlib
import numpy as np
import torch
from tqdm import tqdm

# Attempt to import rotation conversion utilities from PyTorch3D
try:
    from pytorch3d.transforms import axis_angle_to_matrix as aa_to_rotmat_pytorch3d
    from pytorch3d.transforms import matrix_to_rotation_6d as rotmat_to_6d_pytorch3d
    print("Using PyTorch3D for axis-angle to 6D rotation conversion for pose_body.")
except ImportError:
    print("Critical Error: PyTorch3D not found. This script requires PyTorch3D for rotation conversions.")
    print("Please install PyTorch3D: https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md")
    exit()


def convert_pose_body_axis_angle_to_6d(pose_body_aa_tensor, num_body_joints):
    """
    Converts a batch of pose_body axis-angle tensors to 6D rotation representation.
    pose_body_aa_tensor shape: (T, num_body_joints * 3)
    """
    T = pose_body_aa_tensor.shape[0] #

    # Reshape for PyTorch3D: (T * num_body_joints, 3)
    aa_reshaped = pose_body_aa_tensor.reshape(T * num_body_joints, 3) #

    # Convert to rotation matrices: (T * num_body_joints, 3, 3)
    rot_mats = aa_to_rotmat_pytorch3d(aa_reshaped) #

    # Reshape back to per-frame, per-joint matrices: (T, num_body_joints, 3, 3)
    rot_mats_reshaped = rot_mats.reshape(T, num_body_joints, 3, 3) #

    # Convert to 6D representation: (T, num_body_joints, 6)
    pose_body_6d_torch = rotmat_to_6d_pytorch3d(rot_mats_reshaped) #
    return pose_body_6d_torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess BABEL TotalCapture: root_orient(aa), pose_body(6D), trans, betas. Saves train/val/test separately."
    )
    parser.add_argument(
        "--babel_root",
        type=pathlib.Path,
        default="./data/babel_v1.0_release",
        help="Path to the BABEL dataset release directory.",
    )
    parser.add_argument(
        "--amass_root",
        type=pathlib.Path,
        default="./data/amass",
        help="Path to the AMASS dataset directory.",
    )
    parser.add_argument(
        "--save_dir",
        type=pathlib.Path,
        default="./data/babel_processed/totalcapture_body_aa_root_splits", # Adjusted save_dir for clarity
        help="Directory to save the processed .npz files.",
    )
    parser.add_argument(
        "--sample_rate",
        type=int,
        default=1,
        help="Time down-sampling interval (frames). Default is 1.",
    )
    parser.add_argument(
        "--num_betas",
        type=int,
        default=16, # Defaulting to 16 as it's common in AMASS for SMPL-H/X
        help="Number of beta parameters to keep.",
    )
    return parser.parse_args()


def process_babel_sequences(args):
    babel_splits_to_process = ["train", "val", "test"] # Define which splits to process
    target_dataset_name = "TotalCapture"
    NUM_BODY_JOINTS = 21 # SMPL-H body joints (excluding root, hands)

    args.save_dir.mkdir(parents=True, exist_ok=True) #

    for split in babel_splits_to_process:
        # Initialize lists for each split
        all_root_orient_aa = []
        all_pose_body_6d = []
        all_trans = []
        all_betas = []
        all_genders = []
        all_babel_sids = []
        all_frame_counts = []

        babel_json_path = args.babel_root / f"{split}.json"
        if not babel_json_path.exists():
            print(f"Warning: BABEL JSON file not found: {babel_json_path}. Skipping split '{split}'.") #
            continue

        print(f"Loading BABEL '{split}' split from {babel_json_path}...") #
        with open(babel_json_path, "r") as f: #
            babel_data = json.load(f) #

        for babel_sid, seq_info in tqdm(
            babel_data.items(), desc=f"Processing {split} split"
        ): #
            feat_p_str = seq_info.get("feat_p") #
            if not feat_p_str: #
                continue

            if not feat_p_str.startswith(target_dataset_name + "/"): #
                continue

            amass_npz_path = args.amass_root / feat_p_str #
            if not amass_npz_path.exists(): #
                continue

            try:
                bdata = np.load(amass_npz_path, allow_pickle=True) #
            except Exception as e:
                print(f"Error loading AMASS .npz file {amass_npz_path}: {e}. Skipping.") #
                continue

            poses_aa_np = bdata.get("poses") #
            trans_np = bdata.get("trans") #
            betas_np = bdata.get("betas") #

            if "gender" in bdata: #
                gender_val = bdata["gender"] #
                gender = str(gender_val.item()) if hasattr(gender_val, 'item') else str(gender_val) #
            else:
                gender = "neutral" #

            if poses_aa_np is None or trans_np is None or betas_np is None: #
                continue

            if poses_aa_np.shape[1] != 156: # Specific to SMPL-H from AMASS
                # print(f"Warning: poses_aa_np for SID {babel_sid} has unexpected dimension {poses_aa_np.shape[1]} (expected 156 for SMPL-H). Skipping.")
                continue

            betas_np = betas_np[:args.num_betas] #
            if betas_np.shape[0] < args.num_betas: #
                 betas_np = np.pad(betas_np, (0, args.num_betas - betas_np.shape[0]), 'constant') #

            if args.sample_rate > 1: #
                poses_aa_np = poses_aa_np[:: args.sample_rate] #
                trans_np = trans_np[:: args.sample_rate] #

            if poses_aa_np.shape[0] == 0: #
                continue

            T = poses_aa_np.shape[0] #

            # Extract root_orient (axis-angle) and pose_body (axis-angle)
            current_root_orient_aa_np = poses_aa_np[:, :3]             # (T, 3) - Keep as NumPy array
            current_pose_body_aa_torch = torch.from_numpy(poses_aa_np[:, 3:66]).float() # (T, 21*3)

            # Convert only pose_body to 6D
            try:
                current_pose_body_6d = convert_pose_body_axis_angle_to_6d(
                    current_pose_body_aa_torch, NUM_BODY_JOINTS
                ) # Output: (T, 21, 6)
            except Exception as e:
                print(f"Error during 6D conversion for pose_body (SID {babel_sid}): {e}. Skipping.") #
                continue

            all_root_orient_aa.append(current_root_orient_aa_np) # Store as NumPy array
            all_pose_body_6d.append(current_pose_body_6d.cpu().numpy()) #
            all_trans.append(trans_np) #
            all_betas.append(betas_np) #
            all_genders.append(gender) #
            all_babel_sids.append(babel_sid) #
            all_frame_counts.append(T) #

        # Save the processed data for the current split
        if not all_babel_sids: #
            print(f"No sequences from '{target_dataset_name}' found or processed for split '{split}'.") #
            continue

        output_filename = (
            f"babel_totalcapture_{split}_rootAA_body6D_trans_betas" # Added split to filename
            f"_sr{args.sample_rate}_betas{args.num_betas}.npz"
        ) #
        output_path = args.save_dir / output_filename #

        np.savez_compressed(
            output_path,
            root_orient_aa=np.array(all_root_orient_aa, dtype=object), # Saved as axis-angle
            pose_body_6d=np.array(all_pose_body_6d, dtype=object),   # Saved as 6D
            trans=np.array(all_trans, dtype=object), #
            betas=np.array(all_betas, dtype=object), #
            gender=np.array(all_genders), #
            babel_sid=np.array(all_babel_sids), #
            frame_counts=np.array(all_frame_counts), #
            note=(
                f"Processed BABEL {split} data for TotalCapture sub-dataset. " #
                f"root_orient is in axis-angle format (T, 3). " #
                f"pose_body is in 6D rotation format (T, {NUM_BODY_JOINTS}, 6). " #
                f"Sample rate: {args.sample_rate}. Number of betas: {args.num_betas}." #
            ),
        )
        print(f"\nProcessed data for {len(all_babel_sids)} TotalCapture sequences from split '{split}' saved to {output_path}") #

if __name__ == "__main__":
    args = parse_args()
    process_babel_sequences(args)