import os
import numpy as np
from tqdm import tqdm
import torch
from torch import Tensor
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 仅在3D散点图时需要

# 若你的项目结构不同，请修改这两个函数的导入方式
from utils.data_utils import sixd_to_rotmat_torch, convert_rot_mats_to_rot_vecs

if __name__ == "__main__":
    actions = ['posing', 'greeting', 'sitting', 'walking', 'smoking', 'walkingtogether', 'phoning', 'walkingdog',
               'waiting', 'eating', 'discussion', 'purchases', 'sittingdown', 'directions', 'takingphoto', ]
    data_dir = './data/h3.6m'
    path_to_data = os.path.join(data_dir, 'h36m_val_75.npz')
    data = np.load(path_to_data, allow_pickle=True)

    # 假设 all_sixd_data 形状为 (N, T, V, 6)
    all_sixd_data = data['sampled_sixd_seq']
    label_seqs = data['label_seq']
    N, V= all_sixd_data.__len__(),all_sixd_data[0].shape[1]

    # 指定输出文件夹，不存在则创建
    output_folder = './compare_plots'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 1. 任意选取 10 个样本索引 (如果 N<10 则请改小这个数值)
    num_samples = 20
    assert N >= num_samples, f"数据集中样本数({N})不足 10 个！"
    np.random.seed(42)  # 固定随机种子，方便复现
    sample_indices = np.random.choice(N, size=num_samples, replace=False)

    # 2. 对每个关节做可视化
    #    每个关节输出两张大图: 一张 3D 旋转轴分布, 一张 旋转角直方图
    for j in tqdm(range(V), desc="Processing joints"):
        # ============ (A) 先准备“旋转轴分布”的大图 ============
        #    这里用 2行 x 5列 = 10 个子图，具体布局可自行调整
        fig_axis = plt.figure(figsize=(40, 16))  # 根据需要调节
        fig_axis.suptitle(f"Joint {j} - Axis Distribution (10 samples)", fontsize=16)

        for i, sample_idx in enumerate(sample_indices):
            # 取出该 sample 的 6D 序列: (T, V, 6)

            label_idx = label_seqs[sample_idx]
            sixd_seq = all_sixd_data[sample_idx]  # shape (T, V, 6)
            # 转为 tensor: (1, T, V, 6)
            sixd_seq_torch = Tensor(sixd_seq).unsqueeze(0)
            # 转旋转矩阵: (1, T, V, 3, 3) -> squeeze -> (T, V, 3, 3)
            rotmats = sixd_to_rotmat_torch(sixd_seq_torch).squeeze(0)
            # 转轴角: (T, V, 3)
            rotvecs = convert_rot_mats_to_rot_vecs(rotmats)

            # 当前关节 j 的所有时间帧旋转向量: (T, 3)
            joint_rotvecs = rotvecs[:, j, :]
            alpha = torch.norm(joint_rotvecs, dim=-1)  # (T,)
            eps = 1e-8
            mask = (alpha > eps)
            direction = torch.zeros_like(joint_rotvecs)
            direction[mask] = joint_rotvecs[mask] / alpha[mask].unsqueeze(-1)

            # 创建一个 3D 子图
            ax = fig_axis.add_subplot(4, 5, i+1, projection='3d')
            ax.scatter(direction[mask, 0].numpy(),
                       direction[mask, 1].numpy(),
                       direction[mask, 2].numpy(),
                       s=5)
            ax.set_title(f"{actions[int(label_seqs[sample_idx])]}")
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")

        axis_save_path = os.path.join(output_folder, f"joint_{j}_axis.png")
        plt.tight_layout()
        plt.savefig(axis_save_path,dpi=600)
        plt.close(fig_axis)

        # ============ (B) 再准备“旋转角分布”的大图 ============
        fig_angle = plt.figure(figsize=(20, 8))
        fig_angle.suptitle(f"Joint {j} - Angle Distribution (10 samples)", fontsize=16)

        for i, sample_idx in enumerate(sample_indices):
            # 同样流程，取第 sample_idx 个样本
            sixd_seq = all_sixd_data[sample_idx]  # (T, V, 6)
            sixd_seq_torch = Tensor(sixd_seq).unsqueeze(0)
            rotmats = sixd_to_rotmat_torch(sixd_seq_torch).squeeze(0)
            rotvecs = convert_rot_mats_to_rot_vecs(rotmats)

            # 当前关节 j 的所有时间帧旋转向量
            joint_rotvecs = rotvecs[:, j, :]
            alpha = torch.norm(joint_rotvecs, dim=-1).numpy()  # (T,)

            ax = fig_angle.add_subplot(4, 5, i+1)
            ax.hist(alpha, bins=30)
            ax.set_title(f" {actions[int(label_seqs[sample_idx])]}")
            ax.set_xlabel("Angle (rad)")

        angle_save_path = os.path.join(output_folder, f"joint_{j}_angle.png")
        plt.tight_layout()
        plt.savefig(angle_save_path,dpi=600)
        plt.close(fig_angle)

    print(f"可视化完成！已在 {output_folder} 下输出对比图。")
