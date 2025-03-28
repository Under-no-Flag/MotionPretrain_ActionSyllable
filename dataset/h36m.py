
import os
from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm

from utils.forward_kinematics import sixd_to_xyz_torch

def split_seq(seq,max_len):
    '''

    :param seq: (T,...)
    :param max_len: (T//max_len,max_len,...)
    :return:
    '''
    N=seq.shape[0]//max_len
    seq=seq[:N*max_len]
    seq=seq.reshape((N,max_len,*seq.shape[1:]))
    return seq

class Human36mDataset(Dataset):
    def __init__(self, data_dir,
                 split='train',

                 input_length=50,
                 predicted_length=25,
                 xyz_scale=1,
                 use_tiny_dataset=True,
                 max_seq_len=51,
                 ):
        self.path_to_data = os.path.join(data_dir,f'h36m_{split}_{input_length+predicted_length}.npz')

        self.input_length = input_length
        self.predicted_length = predicted_length
        self.seq_len = self.input_length + self.predicted_length
        self.input_xyz = []
        self.future_xyz = []
        self.label = []
        self.max_seq_len = max_seq_len

        self.all_xyz_data = []
        self.all_sixd_data = []
        self.all_labels = []

        if os.path.exists(self.path_to_data):
            data=np.load(self.path_to_data,allow_pickle=True)
            all_xyz=data['sampled_xyz_seq']
            all_sixd=data['sampled_sixd_seq']
            label_seqs=data['label_seq']
            self.dim_to_use=data['dimensions_to_use']
            self.joint_used=data['joint_to_use']

            self.V=self.joint_used.__len__()
            self.C=all_sixd[0].shape[-1]
            for idx in tqdm(range(all_xyz.shape[0])):
                xyz_seq=all_xyz[idx]
                xyz_seq=split_seq(xyz_seq,max_len=self.max_seq_len)
                self.all_xyz_data+=xyz_seq

                sixd_seq=all_sixd[idx]
                sixd_seq=split_seq(sixd_seq,max_len=self.max_seq_len)
                self.all_sixd_data+=sixd_seq

                label_sel = np.full(sixd_seq.shape[0], label_seqs[idx])
                self.all_labels += label_sel.tolist()




        return


    def __len__(self):
        return len(self.all_sixd_data)

    def __getitem__(self, idx):
        return self.all_sixd_data[idx][:-1,...], self.all_sixd_data[idx][1:,...], self.all_labels[idx],self.all_xyz_data[idx][1:,...]


class HumanVQVAESixDDataSet(Dataset):
    def __init__(self,
                 data_dir,
                 split='train',
                 input_length=50,
                 predicted_length=25,
                 max_seq_len=64,
                 sixd_scale=1.0):
        """
        增加 xyz 的读取和拆分，便于后续计算 MPJPE
        """
        self.path_to_data = os.path.join(data_dir, f'h36m_{split}_{input_length + predicted_length}.npz')
        self.max_seq_len = max_seq_len
        self.sixd_scale = sixd_scale

        self.processed_sixd = []
        self.processed_xyz = []  # 用来存储对应的 3D 坐标
        self.processed_label = []  # 额外保存标签
        if os.path.exists(self.path_to_data):
            data = np.load(self.path_to_data, allow_pickle=True)
            all_sixd = data['sampled_sixd_seq']  # (N_seqs, T, V, C=6)
            all_xyz  = data['sampled_xyz_seq']   # (N_seqs, T, V, 3)
            all_label = data['label_seq']

            # 逐序列处理
            for seq_6d, seq_xyz,seq_label in tqdm(zip(all_sixd, all_xyz,all_label), desc="Processing data"):
                # 根据需要对 6D 做缩放
                seq_6d = seq_6d * self.sixd_scale
                # 拆分为若干片段
                split_seqs_6d  = self._split_seq(seq_6d, self.max_seq_len)
                split_seqs_xyz = self._split_seq(seq_xyz, self.max_seq_len)

                self.processed_sixd.extend(split_seqs_6d)
                self.processed_xyz.extend(split_seqs_xyz)
                self.processed_label+=[seq_label]*split_seqs_6d.__len__()  # 整型

        return
    def __len__(self):
        return len(self.processed_sixd)

    def __getitem__(self, idx):
        """
        返回 6D 数据和对应的 3D 坐标:
            - x_6d: shape (V*C, T), 作为 VQVAE 输入
            - x_xyz: shape (T, V, 3)，用于 MPJPE 计算
        """
        seq_6d  = self.processed_sixd[idx]  # (T, V, 6)
        seq_xyz = self.processed_xyz[idx]   # (T, V, 3)
        label = self.processed_label[idx]  # 整型
        # 转成 PyTorch Tensor
        seq_6d  = seq_6d.float()   # (T, V, 6)
        seq_xyz = seq_xyz.float()  # (T, V, 3)

        seq_6d = seq_6d.view(seq_6d.shape[0], -1)

        return seq_6d, seq_xyz,label

    def _split_seq(self, seq, max_len):
        """
        将原始序列 seq (T, V, C) 切分为若干固定长度的片段
        """
        n_clips = seq.shape[0] // max_len
        # 截断余数
        seq = seq[:n_clips * max_len]
        # 分割
        split_seqs = np.split(seq, n_clips, axis=0)  # 每个 shape=(max_len, V, C)
        return split_seqs

if __name__=="__main__":



    # H36m=Human36mDataset(data_dir='../data/h3.6m',split='val')
    # import torch
    # dataloder = torch.utils.data.DataLoader(H36m, batch_size=32, shuffle=False, num_workers=4)
    # for i, (input_sixd, future_sixd, label,target_xyz) in enumerate(dataloder):
    #     print(input_sixd.shape, future_sixd.shape, label.shape)
    #
    #
    #
    #     # test sixd_to_xyz_torch
    #     temp_xyz = sixd_to_xyz_torch(future_sixd)
    #     print(temp_xyz.shape,target_xyz.shape)
    #     if torch.allclose(temp_xyz, target_xyz, rtol=1e-3, atol=1e-3):
    #         print("temp_xyz 和 input_xyz 在阈值范围内相等")
    #     else:
    #         print("temp_xyz 和 input_xyz 存在超过阈值的差异")
    #
    #     # print(temp_xyz[0,0,:],input_xyz[0,0,:])
    #     break

    # 测试用例
    dataset = HumanVQVAESixDDataSet(data_dir='../data/h3.6m', split='train')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    # 验证数据形状
    sample,_ ,_= next(iter(dataloader))
    print(f"输入数据形状: {sample.shape}")  # 期望形状 (32, T, V*C)

    # 测试VQ-VAE重构
    from model.vq_vae import HumanVQVAE
    nb_code = 128
    code_dim = 128
    output_emb_width = 128
    down_t = 3
    stride_t = 2
    width = 1024
    depth = 3
    dilation_growth_rate = 3
    activation = 'relu'
    norm = None


    class Args:
        def __init__(self, dataname, quantizer):
            self.dataname = dataname
            self.quantizer = quantizer
            self.commit = 0.02  # 量化器相关参数
            self.gamma = 0.99  # EMA衰减系数
            self.mu = 0.99  # 重置量化器参数（若需要）


    model = HumanVQVAE(
        Args('h36m','ema_reset'),
        nb_code=nb_code,
        code_dim=code_dim,
        output_emb_width=output_emb_width,
        down_t=down_t,
        stride_t=stride_t,
        width=width,
        depth=depth,
        dilation_growth_rate=dilation_growth_rate,
        activation=activation,
        norm=norm
    )  # 需要定义args参数
    reconstructed, _, _ = model(sample)
    print(f"重构输出形状: {reconstructed.shape}")  # 应与输入形状一致
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(sample,))

    print('FLOPs:', flops / 1e6, 'M')  # 将FLOPs转换为G（十亿）
    print('Params:', params / 1e6, 'M')  # 将参数量转换为M（百万）