import numpy as np
import torch
import torch.nn.functional as F

# ====== 引入已有的工具函数 ======
# 如果你文件结构不同，需根据自己的工程实际情况去改 import。
# 例如:
#   from utils.forward_kinematics import fkl_torch, sixd_to_xyz_torch
#   from utils.data_utils import _some_variables_h36m, rotmat_to_6d, sixd_to_rotmat_torch
from utils.forward_kinematics import fkl_torch
from utils.data_utils import _some_variables_h36m, rotmat_to_6d, sixd_to_rotmat_torch,readCSVasFloat

def partial_sixd_to_xyz_torch(
    sixd: torch.Tensor,
    parent: np.ndarray,
    offset: np.ndarray,
    joint_to_use: np.ndarray
):
    """
    只使用部分关节的 6D 表示(对于不在 joint_to_use 里的关节，视作恒等旋转)，
    前向运动学后返回 joint_to_use 对应的 xyz。

    参数:
        sixd: 形状 (B, T, V_used, 6) 的 6D 表示, 只包含 joint_to_use 对应的关节
        parent, offset: 来自 _some_variables_h36m() 的完整父骨骼索引和骨骼偏移
        joint_to_use: 需要使用(或恢复)的关节索引, 长度 = V_used

    返回:
        xyz_partial: 形状 (B, T, V_used, 3)，对应 joint_to_use 这部分关节的 xyz 坐标
    """
    device = sixd.device

    # 将 6D 转成旋转矩阵 (B,T,V_used,3,3)
    rotmats_used = sixd_to_rotmat_torch(sixd)  # shape=(B,T,V_used,3,3) #已检查 ，无误

    B, T, V_used = rotmats_used.shape[:3]
    J_full = offset.shape[0]  # 全部关节总数(=32)

    # ---- 1) 构造一个 (B, T, J_full, 3, 3) 的大旋转矩阵，并全部初始化为单位阵 ----
    R_full = torch.eye(3, device=device).view(1,1,1,3,3).repeat(B, T, J_full, 1, 1)  # (B,T,J_full,3,3)

    # 将部分关节的旋转矩阵替换进去
    for i, jidx in enumerate(joint_to_use):
        R_full[:, :, jidx, :, :] = rotmats_used[:, :, i, :, :]
    R_full[:, :, 24, :, :] = R_full[:, :, 13, :, :]
    # ---- 2) 前向运动学：从 root(假设下标0) 开始往下累乘 ----
    #      p3d_full 用于存储所有关节的世界坐标 (B,T,J_full,3)，先初始化成 offset

    p3d_full = torch.from_numpy(offset).float().to(device).unsqueeze(0).unsqueeze(0)  # (1,1,J_full,3)
    p3d_full = p3d_full.repeat(B, T, 1, 1)  # (B,T,J_full,3)

    for i in range(1, J_full):
        p = parent[i]


        if p > 0:
            # R_full[:, :, i] = R_full[:, :, i] @ R_full[:, :, p]
            # R_full[:, :, i] = torch.matmul(R_full[:, :, i], R_full[:, :, p])

            # p3d_full[:, :, i] = (offset[i] @ R_full[:, :, p]) + p3d_full[:, :, p]
            offset_i = p3d_full[:, :, i]  # shape=(B,T,3)，先把它当作局部坐标(注：这里和 h36m 原代码保持一致)
            offset_i_world = torch.matmul(
                offset_i.unsqueeze(-2),   # -> (B,T,1,3)
                R_full[:, :, p]          # -> (B,T,3,3)
            ).squeeze(-2)                # -> (B,T,3)
            p3d_full[:, :, i] = offset_i_world + p3d_full[:, :, p]

    # ---- 3) 最后仅取 joint_to_use 关节的 xyz 输出 ----
    xyz_partial = p3d_full[:, :, joint_to_use, :].clone()  # (B,T,V_used,3)
    return xyz_partial

def verify_h36m_prepost_with_partial():
    """
    演示如何在同一个脚本:
      1) 对完整 32 个关节做 fkl_torch 得到旋转矩阵和 xyz
      2) 取其中部分关节转换为 6D 并保存
      3) 从部分关节 6D 表示恢复旋转矩阵并做前向运动学
      4) 对比这部分关节的 xyz 与原先是否一致
    """
    # ============== 1) 模拟原始关节角数据 ==============
    # 假设每帧 99 维(包含 32 个关节 + 根平移旋转)
    N = 5
    # raw_angle = torch.randn(N, 99)
    raw_angle = torch.from_numpy(readCSVasFloat(r"D:\MotionPretrain\data\h36m\S5\directions_2.txt"))
    # 如果要模拟“去掉全局位移和旋转”，就清零:
    raw_angle[:, 0:6] = 0.0

    # ============== 2) 做完整骨骼的 fkl_torch 得到 rots_seq、xyz_seq ==============
    parent, offset, rotInd, expmapInd = _some_variables_h36m()
    # parent[24]=

    with torch.no_grad():
        xyz_seq, rots_seq = fkl_torch(raw_angle, parent, offset, rotInd, expmapInd)
        # rots_seq shape = (N, 32, 3, 3)
    # ============== 3) 在完整骨骼上选取部分关节 ==============
    # 这里演示 22 个关节, 你可以更换

    #调顺序的问题
    # joint_to_use = np.array([1,2,3,4,6,7,8,9,12,13,14,15,25,26,27,29,30,17,18,19,21,22])
    # joint_to_use = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,25,26,27,29,30,17,18,19,21,22])
    joint_to_use = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,25,26])
    # joint_to_use = np.arange(0,32)  #无误
    # joint_to_use = np.arange(1,30)  #无误

    # 取出对应关节的旋转矩阵, 做成 6D 并视为“保存下来”
    rots_seq_partial = rots_seq[:, joint_to_use, :, :]        # shape (N, V_used, 3, 3)
    rots_6d_partial  = rotmat_to_6d(rots_seq_partial)         # shape (N, V_used, 6)

    # 对应关节的 xyz 也取出来, 供后面做“对比”
    xyz_partial_gt = xyz_seq[:, joint_to_use, :]              # shape (N, V_used, 3)

    # ============== 4) 后处理：用部分关节的 6D -> 旋转矩阵 -> FK -> xyz ==============
    # 如果要模拟实际文件读写，可以 np.savez 后再 np.load，这里直接用内存
    saved_6d = rots_6d_partial.clone()

    # sixd_to_rotmat_torch 需要形状 (B,T,V,6)，故扩展维度
    B, T, V_used = 1, N, len(joint_to_use)
    partial_6d_btv6 = saved_6d.unsqueeze(0)  # =>(1,N,V_used,6)
    with torch.no_grad():

        recovered_rotmat= sixd_to_rotmat_torch(partial_6d_btv6)
        xyz_partial_recovered = partial_sixd_to_xyz_torch(
            partial_6d_btv6, parent, offset, joint_to_use
        )
        # 最后 squeeze 回 (N,V_used,3)
        xyz_partial_recovered = xyz_partial_recovered.squeeze(0)

    rotation_diff = (rots_seq_partial - recovered_rotmat).abs()
    max_error = rotation_diff.max().item()

    print(f"[verify] 原始 fkl_torch 得到的旋转矩阵 rots_seq.shape={rots_seq_partial.shape}")
    print(f"[verify] 从 6D 恢复的旋转矩阵 recovered_rotmat.shape={recovered_rotmat.shape}")
    print(f"[verify] 二者的最大绝对误差 = {max_error:.6e}")

    # 根据需要决定阈值，这里随意设 1e-5 ~ 1e-6
    if max_error < 1e-5:
        print("[verify] 旋转矩阵重建结果与预处理非常吻合！")
    else:
        print("[verify] 旋转矩阵重建与预处理不一致，需要检查流程。")

    # ============== 5) 比较 xyz_partial_recovered vs xyz_partial_gt ==============
    diff = (xyz_partial_recovered - xyz_partial_gt).abs()
    max_error = diff.max().item()

    print(f"[verify] joint_to_use={joint_to_use}")
    print(f"[verify] xyz_partial_gt.shape={xyz_partial_gt.shape}, xyz_partial_recovered.shape={xyz_partial_recovered.shape}")
    print(f"[verify] 二者的最大绝对误差 = {max_error:.6e}")

    if max_error < 1e-3:
        print("[verify] 部分关节的 xyz 与原先匹配，误差在可接受范围内！")
    else:
        print("[verify] 差异较大，需要检查流程。")

    # print(xyz_partial_gt[0, :, :])
    # print(xyz_partial_recovered[0, :, :])

if __name__ == "__main__":
    verify_h36m_prepost_with_partial()
