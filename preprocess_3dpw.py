#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-process 3DPW to save SMPL parameters similar to preprocess_babel.py:
  - global_orient (axis-angle)
  - body_pose (6D rotation representation for 23 joints)
  - transl
  - betas
  - gender
"""
import argparse
import os
import pickle
import glob
from pathlib import Path
import numpy as np
import torch
# smplx.lbs.batch_rodrigues is used for axis-angle to rotation matrix conversion
from smplx.lbs import batch_rodrigues
from tqdm import tqdm


# Definition for rotmat_to_6d, assuming it's (..., N, 3, 3) -> (..., N, 6)
# For body_pose, R will be (F, 23, 3, 3)
def rotmat_to_6d(R_tensor):
    """Converts rotation matrices to 6D representation (first two rows)."""
    return R_tensor[..., :2, :].reshape(*R_tensor.shape[:-2], 6)


def parse_args():
    pa = argparse.ArgumentParser(description="Preprocess 3DPW to save SMPL parameters.")
    pa.add_argument('--seq_root', default='./data/3dpw/sequenceFiles',
                    help='Unzipped 3DPW/sequenceFiles directory path.')
    pa.add_argument('--split', default="test", choices=["train", "validation", "test"],
                    help="Dataset split to process.")
    pa.add_argument('--save_dir', default='./data/3dpw/processed_params',
                    help='Output .npz save directory.')
    pa.add_argument('--sample_rate', type=int, default=2,
                    help='Time down-sampling interval (frames).')
    pa.add_argument('--max_seq_len', type=int, default=-1,
                    help='Maximum sequence length for truncation (-1 for no truncation).')
    # num_betas can be an argument if needed, but 3DPW typically uses 10 for SMPL
    # pa.add_argument('--num_betas', type=int, default=10, help="Number of beta parameters to keep.")
    return pa.parse_args()


def to_ndarray(x, dtype=np.float32):
    """Converts input to a numpy ndarray of a specific dtype."""
    if isinstance(x, list):
        if len(x) > 0:
            # Assuming the list contains arrays, and we take the first one
            # This was relevant for data['betas'][0] previously.
            # For poses/trans, they are usually direct ndarrays in 3DPW pkls.
            x_arr = np.asarray(x[0], dtype=dtype)
            if x_arr.ndim == 0 and isinstance(x[0], np.ndarray):  # handles list of 0-dim array with actual array inside
                x_arr = np.asarray(x[0].item(), dtype=dtype)
            x = x_arr
        else:
            return np.array([], dtype=dtype)
    elif isinstance(x, np.ndarray):
        x = x.astype(dtype)
    else:
        x = np.asarray(x, dtype=dtype)
    return x


def process_seq_to_params(pkl_path, sample_rate, max_len, num_betas_to_keep=10):  # Added num_betas_to_keep
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    poses_np = to_ndarray(data['poses'])
    transl_np = to_ndarray(data['trans'])

    # Handle betas, ensuring we take the first 10 from the first actor if it's a list
    raw_betas = data['betas']
    if isinstance(raw_betas, list) and len(raw_betas) > 0:
        betas_np = to_ndarray(raw_betas[0][:num_betas_to_keep])
    elif isinstance(raw_betas, np.ndarray):
        betas_np = to_ndarray(raw_betas[:num_betas_to_keep])
    else:
        print(f"Warning: Unexpected type for betas in {pkl_path}: {type(raw_betas)}. Skipping.")
        return None

    if betas_np.shape[0] < num_betas_to_keep:  # Pad if fewer than expected betas
        betas_np = np.pad(betas_np, (0, num_betas_to_keep - betas_np.shape[0]), 'constant')

    # --- MODIFIED GENDER HANDLING ---
    gender_data = data['genders']
    gender_str = "neutral"  # Default gender

    if isinstance(gender_data, list):
        if len(gender_data) > 0:
            g = str(gender_data[0]).lower()  # Take the first gender, ensure it's a string, convert to lowercase
            if g == 'm' or g == 'male':
                gender_str = 'male'
            elif g == 'f' or g == 'female':
                gender_str = 'female'
            else:
                print(f"Warning: Unknown gender '{g}' in list for {pkl_path}. Defaulting to 'neutral'.")
        else:
            print(f"Warning: Empty gender list for {pkl_path}. Defaulting to 'neutral'.")
    elif isinstance(gender_data, bytes):  # Handle old format if it exists
        g = gender_data.decode('utf-8').lower()
        if g == 'm' or g == 'male':
            gender_str = 'male'
        elif g == 'f' or g == 'female':
            gender_str = 'female'
        else:
            print(f"Warning: Unknown gender '{g}' (bytes) for {pkl_path}. Defaulting to 'neutral'.")
    elif isinstance(gender_data, str):  # Handle if it's already a string
        g = gender_data.lower()
        if g == 'm' or g == 'male':
            gender_str = 'male'
        elif g == 'f' or g == 'female':
            gender_str = 'female'
        else:
            print(f"Warning: Unknown gender '{g}' (str) for {pkl_path}. Defaulting to 'neutral'.")
    else:
        print(f"Warning: Unexpected gender type ({type(gender_data)}) for {pkl_path}. Defaulting to 'neutral'.")
    # --- END OF MODIFIED GENDER HANDLING ---

    poses_sampled_np = poses_np[::sample_rate]
    transl_sampled_np = transl_np[::sample_rate]

    if max_len > 0:
        poses_sampled_np = poses_sampled_np[:max_len]
        transl_sampled_np = transl_sampled_np[:max_len]

    F = poses_sampled_np.shape[0]
    if F == 0:
        return None

    poses_torch = torch.from_numpy(poses_sampled_np).float()

    global_orient_aa = poses_torch[:, :3]
    body_pose_aa_flat = poses_torch[:, 3:]

    num_body_joints = body_pose_aa_flat.shape[1] // 3
    if num_body_joints != 23:
        print(
            f"Warning: Unexpected number of body pose parameters ({body_pose_aa_flat.shape[1]}) in {pkl_path}. Expected 69 for 23 joints. Skipping.")
        return None

    body_pose_aa_reshaped = body_pose_aa_flat.reshape(F * num_body_joints, 3)
    body_pose_rotmats_flat = batch_rodrigues(body_pose_aa_reshaped)
    body_pose_rotmats = body_pose_rotmats_flat.reshape(F, num_body_joints, 3, 3)
    body_pose_6d = rotmat_to_6d(body_pose_rotmats)

    return (
        global_orient_aa.cpu().numpy(),
        body_pose_6d.cpu().numpy(),
        transl_sampled_np,
        betas_np,  # Shape (num_betas_to_keep,)
        gender_str,
        F
    )


def main():
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    all_global_orient_aa = []
    all_body_pose_6d = []
    all_transl = []
    all_betas = []
    all_genders = []
    all_seq_names = []
    all_frame_counts = []

    num_betas_to_keep = 10  # Standard SMPL betas for 3DPW

    split_folder = args.split
    pkl_files_pattern = os.path.join(args.seq_root, split_folder, '*.pkl')
    pkl_files = sorted(glob.glob(pkl_files_pattern))

    if not pkl_files:
        print(
            f"No .pkl files found in {os.path.join(args.seq_root, split_folder)}. Please check --seq_root and --split.")
        return

    for pkl_path in tqdm(pkl_files, desc=f'Processing 3DPW {args.split} sequences'):
        try:
            processed_data = process_seq_to_params(pkl_path, args.sample_rate, args.max_seq_len, num_betas_to_keep)
            if processed_data is None:
                continue

            global_orient_aa, body_pose_6d, transl, betas, gender, frame_count = processed_data
            seq_name = Path(pkl_path).stem

            all_global_orient_aa.append(global_orient_aa)
            all_body_pose_6d.append(body_pose_6d)
            all_transl.append(transl)
            all_betas.append(betas)  # Appending (num_betas_to_keep,) array
            all_genders.append(gender)
            all_seq_names.append(seq_name)
            all_frame_counts.append(frame_count)

        except Exception as e:
            print(f"Error processing {pkl_path}: {e}. Skipping.")
            import traceback
            traceback.print_exc()
            continue

    if not all_seq_names:
        print(f"No sequences were successfully processed for split '{args.split}'. Output file will not be created.")
        return

    global_orient_aa_arr = np.array(all_global_orient_aa, dtype=object)
    body_pose_6d_arr = np.array(all_body_pose_6d, dtype=object)
    transl_arr = np.array(all_transl, dtype=object)

    betas_arr = np.array(all_betas)  # Should now be (N_seqs, num_betas_to_keep)
    genders_arr = np.array(all_genders)
    seq_names_arr = np.array(all_seq_names)
    frame_counts_arr = np.array(all_frame_counts)

    num_body_joints_val = 23

    output_filename = (
        f"3dpw_{args.split}_rootAA_body6D_transl_betas"
        f"_sr{args.sample_rate}_betas{num_betas_to_keep}.npz"  # Added betas count to filename
    )
    out_path = os.path.join(args.save_dir, output_filename)

    np.savez_compressed(
        out_path,
        global_orient_aa=global_orient_aa_arr,
        body_pose_6d=body_pose_6d_arr,
        transl=transl_arr,
        betas=betas_arr,
        gender=genders_arr,
        seq_name=seq_names_arr,
        frame_counts=frame_counts_arr,
        num_body_joints=np.int32(num_body_joints_val),
        num_betas=np.int32(num_betas_to_keep),
        note=(
            f"Processed 3DPW {args.split} data. "
            f"global_orient is axis-angle (T, 3). "
            f"body_pose is 6D rotation (T, {num_body_joints_val}, 6). "
            f"transl is (T, 3). betas is ({num_betas_to_keep},). "
            f"Sample rate: {args.sample_rate}."
        )
    )
    print(f'\nSaved {len(all_seq_names)} processed sequences to {out_path}')


if __name__ == '__main__':
    main()