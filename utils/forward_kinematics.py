import numpy as np
from torch.autograd.variable import Variable
import torch
from utils import data_utils

def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R

def fkl(angles, parent, offset, rotInd, expmapInd):
    """
    Convert joint angles and bone lenghts into the 3d points of a person.
    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L14
    which originaly based on expmap2xyz.m, available at
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m
    Args
      angles: 99-long vector with 3d position and 3d joint angles in expmap format
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    Returns
      xyz: 32x3 3d points that represent a person in 3d space
      rots: 32x3x3 rotation matrices for each joint
    """

    assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        # if not rotInd[i]:  # If the list is empty
        #     xangle, yangle, zangle = 0, 0, 0
        # else:
        #     xangle = angles[rotInd[i][0] - 1]
        #     yangle = angles[rotInd[i][1] - 1]
        #     zangle = angles[rotInd[i][2] - 1]
        if i == 0:
            xangle = angles[0]
            yangle = angles[1]
            zangle = angles[2]
            thisPosition = np.array([xangle, yangle, zangle])
        else:
            thisPosition = np.array([0, 0, 0])

        r = angles[expmapInd[i]]

        thisRotation = expmap2rotmat(r)

        if parent[i] == -1:  # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i, :], (1, 3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i, :] + thisPosition).dot(xyzStruct[parent[i]]['rotation']) + \
                                  xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(xyzStruct[parent[i]]['rotation'])

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    # xyz = xyz[:, [0, 2, 1]]
    # xyz = xyz[:,[2,0,1]]

    rots = [xyzStruct[i]['rotation'] for i in range(njoints)]
    rots = np.array(rots)


    return xyz, rots


def fkl_torch(angles, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.
    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = angles.data.shape[0]
    j_n = offset.shape[0]
    p3d = Variable(torch.from_numpy(offset)).float().unsqueeze(0).repeat(n, 1, 1)#.cuda().unsqueeze(0).repeat(n, 1, 1)
    angles = angles[:, 3:].contiguous().view(-1, 3)
    R = data_utils.expmap2rotmat_torch(angles).view(n, j_n, 3, 3)
    R_origin= R.clone()
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
    # return p3d,R_origin
    return p3d,R


def rotmat_to_xyz_torch(rotmat,  parent, offset,joint_to_use):
    """
    根据部分关节的旋转矩阵，计算所有关节的 3D 坐标。

    :param rotmat:      (B, T, V, 3, 3) 只包含 joint_to_use 中关节的旋转矩阵
    :param parent:      (j_n,)          每个关节的父关节索引
    :param offset:      (j_n, 3)        每个关节的参考姿势下的本地偏移
    :param joint_to_use:长度为 V 的关节索引列表
    :return:
        p3d: (B, T, j_n, 3)  所有关节在 3D 世界坐标系中的位置
    """
    device = rotmat.device
    B, T, V = rotmat.shape[:3]
    j_n = offset.shape[0]  # 骨骼完整关节数

    # -------------------------------------------------------------------
    # 1) 准备用于存储 全关节旋转矩阵 R 和 全关节坐标 p3d
    #    R: (B, T, j_n, 3, 3)
    #    p3d: (B, T, j_n, 3)
    # -------------------------------------------------------------------
    R = torch.zeros((B, T, j_n, 3, 3), dtype=torch.float32, device=device)
    # 将未指定的关节旋转设置为单位矩阵
    R[..., 0, 0] = 1.0
    R[..., 1, 1] = 1.0
    R[..., 2, 2] = 1.0

    # offset 是静态参考位姿下，每个关节的本地偏移；这里将其广播到 (B, T) 两个维度
    p3d = torch.from_numpy(offset).float().to(device)  # (j_n, 3)
    p3d = p3d.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)  # (B, T, j_n, 3)

    # -------------------------------------------------------------------
    # 2) 将部分关节的旋转矩阵嵌入到全关节 R
    #    joint_to_use 中的关节用传入的 rotmat 覆盖
    # -------------------------------------------------------------------
    R = rotmat.clone()

    # -------------------------------------------------------------------
    # 3) 前向运动学：从第 1 号关节开始，沿着 parent 逐级更新
    #    （假设 parent[0] 是根关节，或者根关节的 parent[i] ≤ 0）
    # -------------------------------------------------------------------
    for i in range(1, j_n):
        if parent[i] > 0:
            # 累乘旋转：R_i = R_i * R_parent(i)
            # R[:, :, i] = torch.matmul(R[:, :, i], R[:, :, parent[i]])

            # 累积位置：p3d_i = (offset_i * R_parent(i)) + p3d_parent(i)
            offset_i = p3d[:, :, i]  # (B, T, 3)
            # 先把 offset_i 扩展成 (B, T, 1, 3)，与 R[:, :, parent[i]] (B, T, 3, 3) 矩阵相乘
            offset_i_world = torch.matmul(
                offset_i.unsqueeze(-2),  # -> (B, T, 1, 3)
                R[:, :, parent[i]]  # -> (B, T, 3, 3)
            ).squeeze(-2)  # -> (B, T, 3)

            p3d[:, :, i] = offset_i_world + p3d[:, :, parent[i]]

    # p3d 的形状是 (B, T, j_n, 3)，即所有关节的 3D 位置。
    # return p3d[:, :, joint_to_use, :]
    return p3d


def sixd_to_xyz_torch(sixds,joint_to_use=None):

    """
    Convert joint angles and bone lenghts into the 3d points of a person.
    adapted from
    """
    parent, offset, rotInd, expmapInd = data_utils._some_variables_h36m()
    rotmats=data_utils.sixd_to_rotmat_torch(sixds)


    if joint_to_use is None:
        joint_to_use=np.array([1,2,3,4,6,7,8,9,12,13,14,15,25,26,27,29,30,17,18,19,21,22])

    xyz=rotmat_to_xyz_torch(rotmats,parent,offset,joint_to_use)


    return xyz

if __name__ == "__main__":

    # test sixd_to_xyz
    B,T,V=2,3,22
    sixd = torch.randn(B,T,V, 6)
    xyz = sixd_to_xyz_torch(sixd)
    print(xyz.shape)  # (B, T, V, 3)

