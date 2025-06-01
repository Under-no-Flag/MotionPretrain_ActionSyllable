#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-process 3DPW:
  axis-angle (Rodrigues 72D)  ->  xyz joints  &  6-D rotation repr.

Author: you :-)
"""
import argparse, os, pickle, sys, json, glob
from pathlib import Path
import numpy as np
import torch
import smplx  # pip install smplx
from smplx.lbs import batch_rodrigues  # 同 H3.6M 里的 utils
from tqdm import tqdm


def rotmat_to_6d(R):  # (B, J, 3, 3) -> (B, J, 6)
    return R[..., :2].reshape(*R.shape[:-2], 6)


def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument('--seq_root', default='./data/3dpw/sequenceFiles',
                    help='解压后的 3DPW/sequenceFiles 目录')
    pa.add_argument('--smpl_model_path', default="./data/3dpw/smpl_models/SMPL_MALE.pkl",
                    help='SMPL *.pkl 模型文件所在目录')
    pa.add_argument('--split', default="train")
    pa.add_argument('--save_dir', default='./data/3dpw/processed_data', help='输出 .npz 保存处')
    pa.add_argument('--sample_rate', type=int, default=2,
                    help='时间下采样间隔（帧）')
    pa.add_argument('--gender', choices=['male', 'female', 'neutral'],
                    default='neutral', help='若想统一用同一性别模型可手动指定')
    pa.add_argument('--max_seq_len', type=int, default=-1,
                    help='长序列可截断避免 GPU OOM（-1 不截断）')
    return pa.parse_args()


def build_smpl_layer(model_path, gender, batch):
    return smplx.create(model_path, model_type='smpl',
                        gender=gender, batch_size=batch).eval()


def to_ndarray(x):
    """如果 x 是 list，就取第 0 个并转成 np.ndarray；否则原样返回"""
    if isinstance(x, list):
        x = np.asarray(x[0])
    return x


def process_seq(pkl_path, smpl_layer, sample_rate, max_len):
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    poses = torch.from_numpy(to_ndarray(data['poses'])).float()  # (F,72)
    transl = torch.from_numpy(to_ndarray(data['trans'])).float()  # (F,3)
    betas = torch.from_numpy(to_ndarray(data['betas'])).float()[:10]  # (10,)
    # 下采样
    poses = poses[::sample_rate]
    transl = transl[::sample_rate]
    if max_len > 0:
        poses = poses[:max_len]
        transl = transl[:max_len]

    F = poses.shape[0]
    global_orient = poses[:, :3]  # (F, 3)
    body_pose = poses[:, 3:]  # (F, 69)  ← **扁平 23×3**

    # 确保betas维度正确
    betas = betas.unsqueeze(0).repeat(F, 1)  # (F, 10)
    try:
        smpl_out = smpl_layer(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=transl,
            return_full_pose=True
        )
    except Exception as e:
        print(f"Error: {e}")
        print(f"betas: {betas.shape}")
        print(f"body_pose: {body_pose.shape}")
        print(f"global_orient: {global_orient.shape}")
        print(f"transl: {transl.shape}")

    from smplx.lbs import batch_rodrigues
    rotmats = batch_rodrigues(smpl_out.full_pose.reshape(-1, 3)) \
        .view(F, 24, 3, 3)

    joints = smpl_out.joints[:, :24]  # (F, 24, 3)

    sixd = rotmat_to_6d(rotmats)  # (F, 24, 6)

    # 这里可减去 root 平移，也可保持相机坐标。默认减去 pelvis：
    joints_rel = joints - joints[:, :1]  # root-relative

    return joints_rel.cpu().numpy(), sixd.cpu().numpy()


def main():
    args = parse_args()
    Path(args.save_dir).mkdir(parents=True, exist_ok=True)

    # 先用最常见的 neutral 模型；如果想按性别分，可在 loop 内切换
    smpl_layer = build_smpl_layer(args.smpl_model_path,
                                  gender=args.gender, batch=1)

    xyz_list, sixd_list, name_list = [], [], []

    pkl_files = sorted(glob.glob(os.path.join(args.seq_root, args.split, '*.pkl')))
    # print(pkl_files)
    for pkl_path in tqdm(pkl_files, desc='3DPW sequences'):
        try:
            xyz, sixd = process_seq(pkl_path, smpl_layer,
                                    args.sample_rate, args.max_seq_len)
            seq_name = Path(pkl_path).stem
            xyz_list.append(xyz)
            sixd_list.append(sixd)
            name_list.append(seq_name)
        except Exception as e:
            continue

    xyz_arr = np.array(xyz_list, dtype=object)
    sixd_arr = np.array(sixd_list, dtype=object)
    name_arr = np.array(name_list)

    out_path = os.path.join(args.save_dir,
                            f'3dpw_xyz6d_{args.split}_sr{args.sample_rate}.npz')

    # save   ['sampled_sixd_seq']  # (N_seqs, T, V, C=6) ['sampled_xyz_seq']    # (N_seqs, T, V, 3)['label_seq']
    np.savez_compressed(out_path,
                        sampled_sixd_seq=sixd_arr,
                        sampled_xyz_seq=xyz_arr,
                        label_seq=name_arr,
                        joint_num=np.int32(24),
                        note='root-relative xyz, SMPL 24 joints')
    print(f'\nSaved to {out_path}')


if __name__ == '__main__':
    main()
