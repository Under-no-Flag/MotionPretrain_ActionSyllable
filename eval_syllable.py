from __future__ import annotations

import argparse
import os
import math
from pathlib import Path
from typing import Callable, Tuple
from matplotlib import gridspec

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d)
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import random

from utils import vis
# ------------------------------------------------------------------
# Import the model & segmenter (ensure PYTHONPATH includes project root)
# ------------------------------------------------------------------
from model.motion_vqvae import MotionVQVAE        # noqa: E402
from model.MSS import MotionSyllableSegmenter  # noqa: E402
from dataset.h36m import SyllableDataset,collate_fn
def draw_clip(
    xyz: torch.Tensor,
    bounds: list[int],
    save_path: str,
    title: str = "",
    step: int = 3,          # 每 step 帧取 1 帧
    bar_h_ratio: float = .4 # 0.4 行高给配色条
):
    # 1) 可选下采样，防止帧数过多
    xyz = xyz[::step]                       # (T',V,3)
    T = xyz.size(0)
    n_row = 1                             # 只排 2 行骨架；你也可以设 3
    n_col = math.ceil(T / n_row)

    fig = plt.figure(figsize=(n_col * 3, n_row * 3 + 1))  # 比例更方正
    outer = gridspec.GridSpec(
        2, 1, height_ratios=[1. - bar_h_ratio, bar_h_ratio], hspace=.08
    )

    # ── 1. 骨架序列网格 ────────────────────────
    gs_pose = gridspec.GridSpecFromSubplotSpec(
        n_row, n_col, subplot_spec=outer[0], wspace=.02, hspace=.02
    )
    for i in range(T):
        r, c = divmod(i, n_col)
        ax = fig.add_subplot(gs_pose[r, c], projection="3d")
        vis.show3Dpose(xyz[i].numpy(), ax, radius=1.2)   # radius 决定骨架占比
        ax.axis("off")

    # ── 2. 动作音节条 ──────────────────────────
    ax_bar = fig.add_subplot(outer[1])
    color_tab = plt.cm.get_cmap("tab20")(np.linspace(0, 1, len(bounds) - 1))
    for i in range(len(bounds) - 1):
        ax_bar.axvspan(
            bounds[i] / step, bounds[i + 1] / step,
            0, 1, color=color_tab[i], alpha=.9
        )
    ax_bar.set_xlim(0, T)
    ax_bar.axis("off")

    if title:
        fig.suptitle(title, y=1.02)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fake_predict_boundaries(length: int) -> list[int]:
    """️ 临时代码：随机切 4-8 个段；替换成真实模型推理即可"""
    n_seg = random.randint(4, 8)
    cuts = sorted(random.sample(range(5, length - 5), n_seg - 1))
    return [0] + cuts + [length]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data/h3.6m")
    parser.add_argument("--split", default="val")
    parser.add_argument("--weights", type=str, default="./ckpt/vqvae/vq_vae_h36m.pth", help="Path to VQVAE weights")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_vis", type=int, default=8)
    parser.add_argument("--out", default="./vis_syllable")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max_vis_len", type=int, default=500, help="Max length of visualized sequences")
    parser.add_argument("--chunk_len", type=int, default=100)
    parser.add_argument("--min_seg_len", type=int, default=3)
    args = parser.parse_args()


    joint_to_use =[1, 2, 3, 4, 6, 7, 8, 9, 12, 13, 14, 15, 25, 26, 27, 29, 30, 17, 18, 19, 21, 22]
    # 1. Dataset & loader
    ds = SyllableDataset(args.data_dir, split=args.split)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # 2. Model & segmenter
    device = torch.device(args.device)
    vqvae = MotionVQVAE(
        n_heads=4,
        num_joints=32,
        in_dim=6,
        n_codebook=8,
        n_e=512,
        e_dim=64,
        hid_dim=64,
        beta=0.25,
        quant_min_prop=1.0,
        n_layers=[0, 10],
        seq_len=64,
    ).to(device)
    if args.weights and Path(args.weights).is_file():
        vqvae.load_state_dict(torch.load(args.weights, map_location=device)['net'])
    vqvae.eval()

    segmenter = MotionSyllableSegmenter(vqvae, device=device)

    vis_dir = Path("vis_out")
    vis_dir.mkdir(exist_ok=True)


    # 2) （可选）加载 VQ-VAE + Segmenter；这里只演示随机边界


    # 3) 遍历可视化
    for idx, (sixd, xyz, valid, label) in enumerate(loader):
        full_len = int(valid.sum())
        xyz_clip = xyz[0, :full_len, joint_to_use]  # (T,V,3)
        # seg_bounds = fake_predict_boundaries(full_len)  # or segmenter.detect...
        # ── 预测音节分段 ────────────────────────────────────────────────
        segments_batch, _ = segmenter.segment(sixd.to(device), valid.to(device))
        seg_pairs = segments_batch[0]  # 当前 batch_size=1
        # seg_pairs = [(0,11),(11,26), … ,(115,120)]

        # 把区间对转换成边界列表 [0,11,26,…,120]
        seg_bounds = [seg_pairs[0][0]] + [e for (_, e) in seg_pairs]

        # ---------- ① 依据 chunk_len 把序列拆块 ----------
        part_id = 0
        start = 0
        while start < full_len:
            end = min(start + args.chunk_len, full_len)
            if end - start < args.min_seg_len:  # 剩余不足阈值则忽略
                break

            # 1) 切局部 xyz
            xyz_part = xyz_clip[start:end]

            # 2) 找落在 [start,end) 内的边界并减去 start → 局部 bounds
            local_bounds = [0]
            for b in seg_bounds[1:-1]:
                if start < b < end:
                    local_bounds.append(b - start)
            local_bounds.append(end - start)

            # 3) 保存
            fname = f"clip{idx:03d}_part{part_id}.png"
            save_path = os.path.join(args.out, fname)
            draw_clip(
                xyz_part.cpu(),
                local_bounds,
                save_path,
                title=f"sample {idx}  part {part_id}  |  label {label.item()}",
                step=3, bar_h_ratio=.3  # <= 自己调
            )

            part_id += 1
            start = end

        # ---------- 可选：限制总可视化样本数 ----------
        if idx >= args.num_vis - 1:
            break

if __name__ == "__main__":
    main()