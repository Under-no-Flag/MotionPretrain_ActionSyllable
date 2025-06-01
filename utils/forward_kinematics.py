import numpy as np
from torch.autograd.variable import Variable
import torch
from utils import data_utils
from human_body_prior.body_model.body_model import BodyModel # New import
import pathlib # Already there
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle # Already there from previous context
import smplx # Ensure smplx is imported
from smplx.lbs import batch_rodrigues # For potential direct use if needed, though smplx.create handles it
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


# Add these imports at the top of your eval_transformer_vqvae.py if not already there
import pathlib  # Already there
from human_body_prior.body_model.body_model import BodyModel  # New import

# from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle # Already there from previous context

# Cache for BodyModel instances to avoid re-initialization
_smplh_models_cache = {}
def get_smplh_model_for_babel(gender_str, smplh_model_base_path, num_betas, device):
    """
    Retrieves or initializes a gender-specific SMPL-H BodyModel.
    Uses a simple cache.
    """
    gender_key = gender_str.lower()
    cache_key = (gender_key, num_betas, str(device))  # Add num_betas and device to cache key

    if cache_key not in _smplh_models_cache:
        model_file_path = pathlib.Path(smplh_model_base_path) / gender_key / "model.npz"
        if not model_file_path.exists():
            # Fallback or error if gendered model is not found
            # Try neutral if specific gender is not found and it makes sense for your setup
            print(
                f"Warning: SMPL-H model file not found: {model_file_path}. Trying neutral if available, or check path.")
            neutral_model_path = pathlib.Path(smplh_model_base_path) / "neutral" / "model.npz"
            if neutral_model_path.exists():
                model_file_path = neutral_model_path
                print(f"Using neutral model: {model_file_path}")
            else:
                raise FileNotFoundError(
                    f"SMPL-H model not found for gender '{gender_key}' at {model_file_path} and neutral model also not found.")

        print(f"Loading SMPL-H model for gender '{gender_key}' (using {model_file_path})...")
        _smplh_models_cache[cache_key] = BodyModel(
            bm_fname=str(model_file_path),
            num_betas=num_betas,  # Ensure this matches the model's expected betas
            model_type='smplh'
        ).to(device).eval()
    return _smplh_models_cache[cache_key]


def sixd_to_xyz_babel(
        poses_6d_combined,  # (B, T, 22, 6) where 0 is root, 1-21 are body
        betas,  # (B, num_betas)
        trans,  # (B, T, 3)
        genders,  # List or array of strings, length B, e.g., ['male', 'female', 'male']
        smplh_model_base_path,  # e.g., './amass/smplh'
        num_betas_model,  # Number of betas the BodyModel expects (e.g., 16)
        device
):
    """
    Converts 6D combined poses (root + body) for BABEL (SMPL-H) to 3D joint coordinates.
    Assumes all samples in the batch 'B' can be processed with their respective gender models.
    If genders are mixed, this function will process them one by one, which can be slow.
    For efficiency, try to batch samples of the same gender.
    """
    B, T, V_combined, C_6d = poses_6d_combined.shape
    assert V_combined == 22, f"Expected 22 combined joints (1 root + 21 body), got {V_combined}"
    assert C_6d == 6, f"Expected 6D pose input, got {C_6d}D"

    # Separate root and body poses from the combined input
    root_orient_6d = poses_6d_combined[:, :, 0:1, :]  # (B, T, 1, 6)
    pose_body_6d = poses_6d_combined[:, :, 1:22, :]  # (B, T, 21, 6)

    # Convert 6D to rotation matrices
    # Input to rotation_6d_to_matrix is (..., 6)
    root_orient_rotmat = rotation_6d_to_matrix(root_orient_6d.reshape(B * T * 1, 6))  # (B*T*1, 3, 3)
    pose_body_rotmat = rotation_6d_to_matrix(pose_body_6d.reshape(B * T * 21, 6))  # (B*T*21, 3, 3)

    # Convert rotation matrices to axis-angle
    # Input to matrix_to_axis_angle is (..., 3, 3)
    root_orient_aa = matrix_to_axis_angle(root_orient_rotmat).reshape(B, T, 3)  # (B, T, 3)
    pose_body_aa = matrix_to_axis_angle(pose_body_rotmat).reshape(B, T, 21 * 3)  # (B, T, 63)

    # BodyModel processes (N, ...) where N is batch of frames.
    # We need to handle potentially different genders in the batch B.
    # This simple version assumes genders in batch are the same or iterates (less efficient).

    all_xyz_joints_list = []
    unique_genders = sorted(list(set(g.lower() for g in genders)))

    for unique_gender_str in unique_genders:
        # Find indices for samples with this gender
        gender_mask = [g.lower() == unique_gender_str for g in genders]
        indices_this_gender = [i for i, is_gender in enumerate(gender_mask) if is_gender]

        if not indices_this_gender:
            continue

        B_gender = len(indices_this_gender)

        # Select data for this gender
        current_root_orient_aa = root_orient_aa[indices_this_gender, :, :]  # (B_gender, T, 3)
        current_pose_body_aa = pose_body_aa[indices_this_gender, :, :]  # (B_gender, T, 63)
        current_trans = trans[indices_this_gender, :, :]  # (B_gender, T, 3)
        current_betas = betas[indices_this_gender, :]  # (B_gender, num_betas)

        # Reshape for BodyModel: (Batch*Time, Features)
        current_root_orient_aa_flat = current_root_orient_aa.reshape(B_gender * T, 3)
        current_pose_body_aa_flat = current_pose_body_aa.reshape(B_gender * T, 21 * 3)
        current_trans_flat = current_trans.reshape(B_gender * T, 3)

        # Betas are static per sequence, BodyModel typically handles broadcasting from (B_gender, num_betas)
        # or expects (B_gender*T, num_betas) if explicitly repeated.
        # Let's try (B_gender, num_betas) and see if BodyModel broadcasts.
        # If not, repeat: current_betas_flat = current_betas.unsqueeze(1).repeat(1, T, 1).reshape(B_gender * T, num_betas_model)
        current_betas_model_input = current_betas  # (B_gender, num_betas) -> BodyModel might need (B_gender*T, num_betas)
        # or it might handle broadcasting (1, num_betas) if B_gender=1
        # For human_body_prior, it's often (1, num_betas) for static betas,
        # or (N, num_betas) where N is the batch size (here B_gender*T)

        # The BodyModel in human_body_prior expects betas to match the first dim of other inputs if not (1, num_betas)
        # So, we might need to repeat betas for each frame if B_gender > 1.
        if B_gender > 0:  # Only proceed if there are samples for this gender
            if current_betas_model_input.shape[0] == B_gender and B_gender != B_gender * T:
                current_betas_model_input = current_betas_model_input.unsqueeze(1).expand(-1, T, -1).reshape(
                    B_gender * T, num_betas_model)

        body_model_instance = get_smplh_model_for_babel(unique_gender_str, smplh_model_base_path, num_betas_model,
                                                        device)

        with torch.no_grad():
            body_model_output = body_model_instance(
                root_orient=current_root_orient_aa_flat,
                pose_body=current_pose_body_aa_flat,
                betas=current_betas_model_input,  # Ensure this matches what body_model expects
                trans=current_trans_flat
                # pose_hand should be omitted or zeroed if model expects it but we don't have it.
                # The BodyModel from human_body_prior can take pose_hand=None or omit it.
            )

        # Output Jtr is (B_gender*T, num_output_joints_smplh, 3), e.g., (B_gender*T, 52, 3)
        xyz_joints_flat_gender = body_model_output.Jtr
        xyz_joints_gender = xyz_joints_flat_gender.reshape(B_gender, T, -1, 3)

        # Store results with original batch indices for reordering
        for i, original_idx in enumerate(indices_this_gender):
            all_xyz_joints_list.append((original_idx, xyz_joints_gender[i]))

    # Reorder to original batch order
    all_xyz_joints_list.sort(key=lambda x: x[0])
    ordered_xyz_joints = torch.stack([item[1] for item in all_xyz_joints_list], dim=0)

    return ordered_xyz_joints  # (B, T, num_output_joints_smplh, 3)


# Cache for SMPL model instances to avoid re-initialization
_smpl_models_cache_3dpw = {}


def get_smpl_model_for_3dpw(gender_str, smpl_model_base_path, device):
    """
    Retrieves or initializes a gender-specific SMPL Model using smplx.create.
    Uses a simple cache. Note: num_betas is intrinsic to the SMPL model file.
    """
    gender_key = gender_str.lower()
    # SMPL models from smplx are usually gender-specific by file, not by a num_betas param at loading
    cache_key = (gender_key, str(device))

    if cache_key not in _smpl_models_cache_3dpw:
        # Construct model path, e.g., ./data/3dpw/smpl_models/SMPL_MALE.pkl
        # Ensure the model path template or logic matches how your SMPL files are stored.
        # The original preprocess_3dpw.py used a single model path from args.
        # Here we make it more flexible if smpl_model_base_path points to a directory.
        model_file = pathlib.Path(smpl_model_base_path) / f"SMPL_{gender_key.upper()}.pkl"

        if not model_file.exists():
            # Fallback to neutral if specific gender model not found and if it's part of your setup.
            # Or raise an error if gender-specific models are required.
            print(f"Warning: SMPL model file not found: {model_file}. Check smpl_model_base_path and gender.")
            # Example fallback (adjust if your neutral model has a different name or path)
            neutral_model_file = pathlib.Path(smpl_model_base_path) / "SMPL_NEUTRAL.pkl"
            if neutral_model_file.exists():
                print(f"Attempting to use neutral model: {neutral_model_file}")
                model_file = neutral_model_file
            else:
                raise FileNotFoundError(
                    f"SMPL model not found for gender '{gender_key}' at {model_file} and neutral model also not found.")

        print(f"Loading SMPL model from: {model_file} for gender '{gender_key}'")
        # batch_size=1 because we process each sample in the batch individually due to gender
        # smplx.create handles num_betas from the model file itself.
        _smpl_models_cache_3dpw[cache_key] = smplx.create(
            model_path=model_file,  # smplx.create expects directory
            model_type='smpl',
            gender=gender_key,  # Pass gender to smplx.create
            # batch_size needs to be T (sequence length) for each sample if processing one sample at a time
            # Or, if we flatten B*T, then it's B*T.
            # Since we iterate per sample in batch B, we can set batch_size for SMPL to T.
            # However, smplx models are often created with batch_size=1 and then inputs are repeated.
            # Let's stick to processing (T, features) per sample.
        ).to(device).eval()
    return _smpl_models_cache_3dpw[cache_key]


def sixd_to_xyz_3dpw(
        poses_6d_combined,  # (B, T, 24, 6) where 0 is root, 1-23 are body for SMPL
        betas,  # (B, num_betas_data, e.g., 10)
        trans,  # (B, T, 3)
        genders,  # List or array of strings, length B, e.g., ['male', 'female']
        smpl_model_base_path,  # e.g., './data/3dpw/smpl_models/' (directory containing SMPL_MALE.pkl etc.)
        device
):
    """
    Converts 6D combined poses for 3DPW (SMPL) to 3D joint coordinates.
    Processes each sample in the batch individually to handle gender.
    """
    B, T, V_combined, C_6d = poses_6d_combined.shape
    assert V_combined == 24, f"Expected 24 combined joints (1 root + 23 body for SMPL), got {V_combined}"
    assert C_6d == 6, f"Expected 6D pose input, got {C_6d}D"

    # These imports should be at the top of the calling script
    from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_axis_angle

    # Separate root and body poses
    root_orient_6d = poses_6d_combined[:, :, 0:1, :]  # (B, T, 1, 6)
    body_pose_6d = poses_6d_combined[:, :, 1:24, :]  # (B, T, 23, 6)

    # Convert 6D to rotation matrices, then to axis-angle
    root_orient_rotmat_flat = rotation_6d_to_matrix(root_orient_6d.reshape(B * T * 1, 6))  # (B*T, 3, 3)
    body_pose_rotmat_flat = rotation_6d_to_matrix(body_pose_6d.reshape(B * T * 23, 6))  # (B*T*23, 3, 3)

    root_orient_aa_flat = matrix_to_axis_angle(root_orient_rotmat_flat)  # (B*T, 3)
    body_pose_aa_flat = matrix_to_axis_angle(body_pose_rotmat_flat)  # (B*T*23, 3)

    # Reshape back to (B, T, Features) or (B*T, Features) as needed by SMPL layer
    # SMPL layer usually takes (N, 3) for global_orient and (N, 69) for body_pose where N is batch_of_frames
    global_orient_for_smpl = root_orient_aa_flat.reshape(B, T, 3)
    body_pose_for_smpl = body_pose_aa_flat.reshape(B, T, 23 * 3)  # (B, T, 69)

    output_xyz_joints_list = [None] * B

    for i in range(B):
        current_gender_str = genders[i]
        smpl_model_instance = get_smpl_model_for_3dpw(current_gender_str, smpl_model_base_path, device)

        # Prepare inputs for this specific sample (all its frames T)
        sample_global_orient = global_orient_for_smpl[i]  # (T, 3)
        sample_body_pose = body_pose_for_smpl[i]  # (T, 69)
        sample_betas = betas[i:i + 1, :]  # (1, num_betas_data), smplx will use its model's num_betas
        # and often expects (T, num_betas) or broadcasts (1, num_betas)
        sample_transl = trans[i]  # (T, 3)

        # smplx.SMPL.forward expects betas to be (batch_size, num_betas).
        # If batch_size for SMPL layer is T, then betas should be (T, num_betas).
        # The original preprocess_3dpw.py did: betas.unsqueeze(0).repeat(F, 1)
        # So, we replicate betas for each frame for this sample.

        # The smpl_layer from original script was created with batch_size=1.
        # If we are passing T frames, we should either create smpl_layer with batch_size=T
        # or pass T,10 for betas.
        # Let's assume the model from cache is generic, and we adapt inputs.
        # It's common for smplx models to broadcast betas if shape is (1, num_betas)
        # and other inputs are (T, ...).

        # However, to be safe and match original preprocess_3dpw's smpl_layer call:
        current_T = sample_global_orient.shape[0]
        smpl_betas_input = sample_betas.repeat(current_T, 1)
        with torch.no_grad():
            smpl_output = smpl_model_instance(
                betas=smpl_betas_input,
                body_pose=sample_body_pose,
                global_orient=sample_global_orient,
                transl=sample_transl,
                return_full_pose=True  # Not strictly needed if only joints are used
            )

        # Output joints from smplx.SMPL are typically (T, 24, 3) for the basic joint set
        output_xyz_joints_list[i] = smpl_output.joints[:, :24, :]  # Ensure we take the 24 SMPL joints

    return torch.stack(output_xyz_joints_list, dim=0)  # (B, T, 24, 3)

if __name__ == "__main__":

    # test sixd_to_xyz
    B,T,V=2,3,22
    sixd = torch.randn(B,T,V, 6)
    xyz = sixd_to_xyz_torch(sixd)
    print(xyz.shape)  # (B, T, V, 3)

