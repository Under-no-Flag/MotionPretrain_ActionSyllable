# ----------------- 新增 import -----------------
import torch.nn.functional as F
from torch.linalg import svd  # torch>=1.9 已支持
import torch
# ----------------- 工具：帧级 Procrustes 对齐 -----------------
def compute_pa_mpjpe(pred_xyz, gt_xyz):
    """
    pred_xyz, gt_xyz: (B, T, V, 3)  or (T, V, 3)  or (V, 3)
    * 逐帧* 做相似变换对齐 (平移+旋转+统一缩放)，
      再计算 L2 误差均值 -> PA-MPJPE
    返回形状与输入 batch 保持一致的 per-frame per-sample 均值 (不做 mm ↔ m 转换)
    """
    if pred_xyz.ndim == 3:  # (T,V,3) => 插 batch 维
        pred_xyz, gt_xyz = pred_xyz[None], gt_xyz[None]

    B, T, V, _ = pred_xyz.shape
    pred_c = pred_xyz - pred_xyz.mean(dim=2, keepdim=True)   # 中心化
    gt_c   = gt_xyz   - gt_xyz.mean(dim=2, keepdim=True)

    # (B,T,3,3)  求最佳旋转
    H = torch.einsum('btvi,btvj->btij', pred_c, gt_c)         # (B,T,3,3)
    U, S, Vh = svd(H)
    R = torch.matmul(Vh.transpose(-2,-1), U.transpose(-2,-1)) # (B,T,3,3)
    # 可能出现反射，修一下
    det_mask = (torch.det(R) < 0).float().view(B, T, 1, 1)
    Vh_fix = torch.cat([Vh[..., :2, :], Vh[..., 2:, :]*(-1)], dim=-2)
    R = torch.where(det_mask.bool(), torch.matmul(Vh_fix.transpose(-2,-1), U.transpose(-2,-1)), R)

    # 统一缩放
    var_pred = (pred_c**2).sum(dim=(-1,-2), keepdim=True)     # (B,T,1,1)
    scale = (S.sum(dim=-1, keepdim=True)/var_pred.squeeze(-1)).unsqueeze(-1)  # (B,T,1,1)

    # 对齐后误差
    pred_aligned = scale * torch.einsum('btvj,btij->btvi', pred_c, R)
    pa_mpjpe_frame = torch.linalg.norm(pred_aligned - gt_c, dim=-1).mean(dim=-1)  # (B,T)
    return pa_mpjpe_frame.mean(dim=-1)  # (B,)



# ----------------- 工具：Acceleration Error -----------------
def compute_acceleration_error(pred_xyz, gt_xyz):
    """
    二阶差分取得加速度：
      a_t = p_{t+1} - 2 p_t + p_{t-1}
    计算 (B,T-2,V,3) 上的 L2 误差均值
    """
    if pred_xyz.shape[1] < 3:   # 帧数不足 3 时返回 nan 以免除零
        return torch.full((pred_xyz.shape[0],), float('nan'), device=pred_xyz.device)

    pred_acc = pred_xyz[:,2:] - 2*pred_xyz[:,1:-1] + pred_xyz[:,:-2]
    gt_acc   = gt_xyz[:,2:]   - 2*gt_xyz[:,1:-1]   + gt_xyz[:,:-2]
    acc_err = torch.linalg.norm(pred_acc - gt_acc, dim=-1).mean(dim=(-1,-2))  # (B,)
    return acc_err
