import torch
import argparse

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import torch
from torch.autograd.variable import Variable
import os

# os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# print(os.getcwd())


from utils.data_utils import _some_variables_h36m,readCSVasFloat,rotmat_to_6d
from utils.forward_kinematics import fkl_torch

def get_parser():

    parser = argparse.ArgumentParser(description='Action Syllable')
    parser.add_argument('--split', type=int, default=0, help='train,val, test')
    parser.add_argument('--save_dir', type=str, default='./data/h3.6m', help='')
    parser.add_argument('--tiny',  action='store_true', help='ceate a small dataset')
    parser.add_argument('--seq_len', type=int, default=75, help='')
    parser.add_argument('--max_seq_len', type=int, default=75, help='max pretrain seq len')


    return parser


def process_skeleton(the_seq):

    parent, offset, rotInd, expmapInd = _some_variables_h36m()
    # print(the_seq.shape)
    xyz_seq,rots_seq= fkl_torch(the_seq,parent,offset,rotInd,expmapInd)
    sixd_seq = rotmat_to_6d(rots_seq)
    return xyz_seq,sixd_seq

if __name__ == "__main__":
    """
    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216
    :param path_to_dataset:
    :param subjects:
    :param actions:
    :param sample_rate:
    :param seq_len:
    :return:
    """
    # 切换到上一级目录
    
    print("------start preprogress human3.6m dataset------")

    args = get_parser().parse_args()

    path_to_dataset = './data/h36m/'
    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]

    subs = [[1, 6, 7, 8, 9], [5], [11]]
    sample_rate = 2
    subjects = subs[args.split]
    seq_len = args.seq_len

    sampled_xyz_seq= []
    sampled_sixd_seq = []
    label_seq = []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            if not (subj == 12):
                # if not (subj == 5):
                for subact in [1, 2]:  # subactions
                    filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)

                    print(f"procesing {filename}")

                    action_sequence = readCSVasFloat(filename)
                    n, d = action_sequence.shape
                    even_list = range(0, n, sample_rate)
                    num_frames = len(even_list)
                    the_sequence = np.array(action_sequence[even_list, :])
                    the_seq = Variable(torch.from_numpy(the_sequence)).float()  # .cuda()
                    # remove global rotation and translation
                    the_seq[:, 0:6] = 0

                    xyz_seq,sixd_seq= process_skeleton(the_seq)



                    label_sel=np.array([action_idx])


                    sampled_xyz_seq.append(xyz_seq)
                    sampled_sixd_seq.append(sixd_seq)
                    label_seq.append(label_sel)



    print(f"sampled_xyz_seq: {sampled_xyz_seq[0].shape}")
    print(f"sampled_sixd_seq: {sampled_sixd_seq[0].shape}")
    print(f"label_seq: {label_seq[0].shape}")
    sampled_xyz_seq = np.array(sampled_xyz_seq,dtype=object)
    sampled_sixd_seq = np.array(sampled_sixd_seq,dtype=object)
    label_seq = np.array(label_seq)

    # 选择了22个关节，待增加2个脚趾关节
    joint_to_use=np.array([1,2,3,4,6,7,8,9,12,13,14,15,25,26,27,29,30,17,18,19,21,22])
    dimensions_to_use = np.concatenate((joint_to_use * 3, joint_to_use * 3 + 1, joint_to_use * 3 + 2))

    #保存路径不存在则创建
    if os.path.exists(args.save_dir) == False:
        os.makedirs(args.save_dir)

    save_prefix = ['train', 'val', 'test']
    save_path = os.path.join(args.save_dir, f'h36m_{save_prefix[args.split]}_{seq_len}.npz')
    np.savez(save_path, sampled_xyz_seq=sampled_xyz_seq, sampled_sixd_seq=sampled_sixd_seq,joint_to_use=joint_to_use,
             dimensions_to_use=dimensions_to_use,label_seq=label_seq)


    print(f"save to {save_path}")

    print("------end preprogress human3.6m dataset------")