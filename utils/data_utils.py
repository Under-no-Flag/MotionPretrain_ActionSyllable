import numpy as np
import torch
from torch.autograd.variable import Variable
import torch.nn.functional as F


def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


# def expmap2_sixdof_torch(expmap):
#     """
#     Convert an exponential map to a 6Dof representation
#     Args
#       expmap: T x 99
#     Returns
#         the 6Dof representation
#         """
#     parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()
#     return rotvec


# some variables of human3.6m dataset
def _some_variables_h36m():
    """
    borrowed from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100
    We define some variables that are useful to run the kinematic tree
    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5,
                       1, 7, 8, 9, 10,
                       1, 12, 13, 14, 15,
                       13,17, 18, 19, 20, 21, 20, 23,
                       13, 25, 26, 27, 28, 29, 28, 31]) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000,
         -132.948591, 0.000000, 0.000000,
         0.000000, -442.894612, 0.000000,
         0.000000,-454.206447, 0.000000,
         0.000000, 0.000000, 162.767078,
         0.000000, 0.000000, 74.999437,
         132.948826, 0.000000,0.000000,
         0.000000, -442.894413,0.000000,
         0.000000, -454.206590,0.000000,
         0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948,
         0.000000, 0.100000, 0.000000,
         0.000000, 233.383263, 0.000000,
         0.000000,257.077681, 0.000000,
         0.000000, 121.134938, 0.000000,
         0.000000, 115.002227, 0.000000,
         0.000000, 257.077681,0.000000,
         0.000000, 151.034226, 0.000000,
         0.000000, 278.882773, 0.000000,
         0.000000, 251.733451, 0.000000,
         0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 99.999627,
         0.000000, 100.000188, 0.000000,
         0.000000,0.000000, 0.000000,
         0.000000, 257.077681, 0.000000,
         0.000000, 151.031437, 0.000000,
         0.000000, 278.892924,0.000000,
         0.000000, 251.728680, 0.000000,
         0.000000, 0.000000, 0.000000,
         0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000,
         0.000000, 0.000000, 0.000000])
    offset = offset.reshape(-1, 3)
    # print(offset.shape)
    rotInd = [[5, 6, 4],
              [8, 9, 7],
              [11, 12, 10],
              [14, 15, 13],
              [17, 18, 16],
              [],
              [20, 21, 19],
              [23, 24, 22],
              [26, 27, 25],
              [29, 30, 28],
              [],
              [32, 33, 31],
              [35, 36, 34],
              [38, 39, 37],
              [41, 42, 40],
              [],
              [44, 45, 43],
              [47, 48, 46],
              [50, 51, 49],
              [53, 54, 52],
              [56, 57, 55],
              [],
              [59, 60, 58],
              [],
              [62, 63, 61],
              [65, 66, 64],
              [68, 69, 67],
              [71, 72, 70],
              [74, 75, 73],
              [],
              [77, 78, 76],
              []]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd

def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = Variable(torch.eye(3, 3).repeat(n, 1, 1)).float() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R

def rotmat_to_6d(R):
    """
    Converts rotation matrix to 6Dof representation
    :param R: T*N*3*3 or N*3*3
    :return: T*N*6 or N*6

  
    """
    if len(R.shape) == 4:
        return R[...,:2].transpose(3,2).reshape(R.shape[0], R.shape[1], -1)
    elif len(R.shape) == 3:
        return R[...,:2].transpose(2,1).reshape(R.shape[0], -1)
    else:
        raise ValueError("Invalid shape of rotation matrix")



def sixd_to_rotmat_torch(sixd):
    """
    Converts a 6D rotation representation to a 3×3 rotation matrix.
    :param sixd: Tensor of shape (B, T, V, 6).
    :return: Tensor of shape (B, T, V, 3, 3) representing rotation matrices.
    """
    B, T, V = sixd.shape[:3]

    # 1) Reshape [B, T, V, 6] -> [B, T, V, 2, 3]
    #    So that we have two 3D vectors per rotation
    reshaped = sixd.view(B, T, V, 2, 3)

    # 2) Split into two vectors, a and b
    a = reshaped[..., 0, :]  # shape: (B, T, V, 3)
    b = reshaped[..., 1, :]  # shape: (B, T, V, 3)

    # 3) First basis vector (x) is the normalized 'a'
    x = F.normalize(a, dim=-1)  # shape: (B, T, V, 3)

    # 4) Make b orthogonal to x, then normalize
    proj = (x * b).sum(dim=-1, keepdim=True) * x
    y = F.normalize(b - proj, dim=-1)  # shape: (B, T, V, 3)

    # 5) z is the cross product of x and y
    z = torch.cross(x, y, dim=-1)      # shape: (B, T, V, 3)

    # 6) Stack x, y, z along the last dimension to get the rotation matrix
    rotmat = torch.stack((x, y, z), dim=-1)  # shape: (B, T, V, 3, 3)
    return rotmat




def verify_so3(rotmat, eps=1e-5):
    """
    验证输出矩阵是否满足 SO(3) 的性质.
    :param rotmat: 形状 (B, T, V, 3, 3) 的旋转矩阵张量
    :param eps: 判断误差的阈值
    """
    # 1) 生成单位矩阵以对比
    eye = torch.eye(3, device=rotmat.device).view(1,1,1,3,3)

    # 2) 检查正交性: R^T R 是否等于单位矩阵
    #    计算与单位矩阵的最大误差
    identity_check = torch.matmul(rotmat.transpose(-1, -2), rotmat)
    ortho_error = (identity_check - eye).abs().max()

    # 3) 检查行列式为 +1
    det_vals = torch.det(rotmat)
    det_error = (det_vals - 1).abs().max()

    print(f"Max orthonormality error = {ortho_error.item():.6f}")
    print(f"Max determinant error    = {det_error.item():.6f}")

    # 4) 综合判定是否满足 SO(3)
    is_orthonormal = ortho_error < eps
    is_det_one = det_error < eps
    is_so3 = is_orthonormal and is_det_one
    print(f"Is SO(3): {is_so3}")
    return is_so3


import torch

def rotation_matrix_to_axis_angle(rot_mats: torch.Tensor) -> torch.Tensor:
    """
    将形状为 (..., 3, 3) 的旋转矩阵转换为旋转向量（轴角表示）
    返回形状为 (..., 3) 的旋转向量
    """
    # 输入检查
    assert rot_mats.shape[-2:] == (3, 3), "最后两个维度必须是 3x3"

    # 计算旋转角度 θ
    tr = rot_mats[..., 0, 0] + rot_mats[..., 1, 1] + rot_mats[..., 2, 2]
    theta = torch.arccos(torch.clamp((tr - 1) / 2, -1.0, 1.0))

    # 处理小角度情况（θ ≈ 0）
    small_theta = theta < 1e-6
    large_theta = ~small_theta

    # 初始化旋转向量
    rot_vecs = torch.zeros_like(rot_mats[..., 0])

    # 情况 1：θ ≈ 0，旋转向量为 [0, 0, 0]
    rot_vecs[small_theta] = 0.0

    # 情况 2：θ ≈ π（需处理奇异情况）
    if large_theta.any():
        # 计算旋转轴方向
        R = rot_mats[large_theta]
        axis = torch.stack([
            R[..., 2, 1] - R[..., 1, 2],
            R[..., 0, 2] - R[..., 2, 0],
            R[..., 1, 0] - R[..., 0, 1]
        ], dim=-1)
        axis_norm = torch.norm(axis, dim=-1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)

        # 计算旋转向量：axis * theta
        rot_vecs[large_theta] = axis * theta[large_theta].unsqueeze(-1)

    return rot_vecs

def convert_rot_mats_to_rot_vecs(rot_mats: torch.Tensor) -> torch.Tensor:
    """
    输入形状： (T, V, 3, 3)
    输出形状： (T, V, 3)
    """
    original_shape = rot_mats.shape
    T, V = original_shape[:2]

    # 展平为 (T*V, 3, 3)
    rot_mats_flat = rot_mats.reshape(-1, 3, 3)

    # 转换为旋转向量
    rot_vecs_flat = rotation_matrix_to_axis_angle(rot_mats_flat)

    # 恢复形状 (T, V, 3)
    rot_vecs = rot_vecs_flat.reshape(T, V, 3)
    return rot_vecs


def get_adjacency_matrix(dataset_name='h36m'):
    """
    获取数据集中使用的关节连接矩阵.
    """
    if dataset_name == 'h36m':
        # H3.6M pairs definition remains unchanged
        pairs= [
            (0, 1), (1, 2), (2, 3), (3, 4),(4,5),
            (0, 6), (6, 7), (7, 8),(8,9),(9, 10),
            (11, 12),(13, 14), (14, 15),
            (16,17), (17,18), (18,19),(20,21),(19,22),
            (24,25), (25, 26), (26, 27),  (28, 29),(27,30)
        ]
        num_joints = 32
        adjacency_matrix = torch.eye(num_joints, dtype=torch.float32)
        for i, j in pairs:
            if i < num_joints and j < num_joints:
                adjacency_matrix[i, j] = 1.0
                adjacency_matrix[j, i] = 1.0
    elif dataset_name == '3dpw':
        # 3DPW (SMPL 24 joints) pairs definition remains unchanged
        pairs = [
            (0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8),
            (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15),
            (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21),
            (20, 22), (21, 23)
        ]
        num_joints = 24
        adjacency_matrix = torch.eye(num_joints, dtype=torch.float32)
        for i, j in pairs:
            if i < num_joints and j < num_joints:
                adjacency_matrix[i, j] = 1.0
                adjacency_matrix[j, i] = 1.0
    elif dataset_name == 'babel': # New entry for BABEL (SMPL-H body without hands)
        # Corresponds to root_orient (1 joint) + pose_body (21 joints) = 22 joints
        # Joint mapping (example, assuming root is 0, then body joints follow SMPL-H order):
        # 0: Pelvis (from root_orient)
        # 1-21: Body joints (from pose_body, e.g., L_Hip, R_Hip, Spine1, ..., L_Wrist, R_Wrist)
        # The pairs below map to this 0-21 indexing for these 22 joints.
        pairs = [
            # Pelvis connections
            (0, 1), (0, 2), (0, 3),  # Pelvis to L_Hip, R_Hip, Spine1
            # Left Leg
            (1, 4), (4, 7), (7, 10), # L_Hip -> L_Knee -> L_Ankle -> L_Foot (L_Ankle's child)
            # Right Leg
            (2, 5), (5, 8), (8, 11), # R_Hip -> R_Knee -> R_Ankle -> R_Foot (R_Ankle's child)
            # Spine
            (3, 6), (6, 9),          # Spine1 -> Spine2 -> Spine3
            # Neck and Head
            (9, 12), (12, 15),       # Spine3 -> Neck -> Head
            # Left Arm (via Collar)
            (9, 13),                 # Spine3 to L_Collar
            (13, 16),                # L_Collar to L_Shoulder
            (16, 18),                # L_Shoulder to L_Elbow
            (18, 20),                # L_Elbow to L_Wrist
            # Right Arm (via Collar)
            (9, 14),                 # Spine3 to R_Collar
            (14, 17),                # R_Collar to R_Shoulder
            (17, 19),                # R_Shoulder to R_Elbow
            (19, 21)                 # R_Elbow to R_Wrist
        ]
        num_joints = 22 # 1 (root) + 21 (body joints)
        adjacency_matrix = torch.eye(num_joints, dtype=torch.float32)
        for i, j in pairs:
            if i < num_joints and j < num_joints:
                adjacency_matrix[i, j] = 1.0
                adjacency_matrix[j, i] = 1.0
    else:
        raise ValueError(f"Unsupported dataset name: {dataset_name}. Please use 'h36m', '3dpw', or 'babel_smplh_body'.")
    return adjacency_matrix

# Example usage:
# adj_matrix_babel = get_adjacency_matrix('babel_smplh_body')
# print("BABEL SMPL-H Body Adjacency Matrix (22 joints):")
# print(adj_matrix_babel)
# print(adj_matrix_babel.shape)

# adj_matrix_3dpw = get_adjacency_matrix('3dpw')
# print("\n3DPW Adjacency Matrix (24 joints):")
# print(adj_matrix_3dpW)
# print(adj_matrix_3dpw.shape)
def get_motion_chains(dataset_name='h36m'):
    """根据数据集名称定义运动链用于可视化"""
    if dataset_name == 'h36m':  # 32 joints
        return [  # 示例 H3.6M 运动链 (你需要根据你的32关节定义调整)
            [0, 1, 2, 3, 4, 5],  # Right Leg
            [0, 6, 7, 8, 9, 10],  # Left Leg
            [0, 11, 12, 13, 14, 15],  # Torso, Neck, Head (adjust 11 to be pelvis/root if 0 is not)
            [12, 16, 17, 18, 19, 20],  # Right Arm (from neck/collar)
            [12, 24, 25, 26, 27, 28]  # Left Arm (from neck/collar)
            # 可能还有其他链，如从肩胛骨开始的手臂等
        ]
    elif dataset_name == '3dpw':  # SMPL 24 joints
        # 0:pelvis, 1:r_hip, 2:l_hip, 3:spine1...
        return [
            [0, 1, 4, 7, 10],  # Right Leg
            [0, 2, 5, 8, 11],  # Left Leg
            [0, 3, 6, 9, 12, 15],  # Spine and Head
            [9, 13, 16, 18, 20, 22],  # Right Arm chain (from upper spine/collar)
            [9, 14, 17, 19, 21, 23]  # Left Arm chain (from upper spine/collar)
        ]
    elif dataset_name == 'babel_smplh_body':  # 22 joints (root + 21 body)
        # 0:Pelvis, 1:L_Hip, 2:R_Hip, 3:Spine1 ... 20:L_Wrist, 21:R_Wrist
        return [
            [0, 2, 5, 8, 11],  # Right Leg (Pelvis -> R_Hip -> R_Knee -> R_Ankle -> R_Foot)
            [0, 1, 4, 7, 10],  # Left Leg (Pelvis -> L_Hip -> L_Knee -> L_Ankle -> L_Foot)
            [0, 3, 6, 9, 12, 15],  # Spine and Head
            [9, 14, 17, 19, 21],  # Right Arm (Spine3 -> R_Collar -> R_Shoulder -> R_Elbow -> R_Wrist)
            [9, 13, 16, 18, 20]  # Left Arm (Spine3 -> L_Collar -> L_Shoulder -> L_Elbow -> L_Wrist)
        ]
    else:
        print(f"Warning: Motion chains not defined for dataset {dataset_name}. Returning empty list.")
        return []

if __name__ == "__main__":

  # test rotmat_to_6d
  # R = torch.randn(10, 3, 3)
  # print("before",R[1])
  # print("after",rotmat_to_6d(R)[1])



  # test sixd_to_rotmat
  sixd = torch.randn(10, 24, 23, 6)
  rand_rot = torch.randn(10, 24, 23, 3,3)
  print("before",sixd[0,0,0,:])
  print("after",sixd_to_rotmat_torch(sixd)[0,0,0,:,:])
  a=sixd_to_rotmat_torch(sixd)[0,0,0,:,0]
  b=sixd_to_rotmat_torch(sixd)[0,0,0,:,1]
  c=sixd_to_rotmat_torch(sixd)[0,0,0,:,2]

  verify_so3(sixd_to_rotmat_torch(sixd))
  verify_so3(rand_rot)