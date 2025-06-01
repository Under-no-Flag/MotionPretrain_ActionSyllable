
import os
from typing import Callable

from torch.utils.data import Dataset
import numpy as np
import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

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
        self.path_to_data = os.path.join(data_dir,f'h36m_{split}.npz')

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
        self.path_to_data = os.path.join(data_dir, f'h36m_{split}.npz')
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

        # seq_6d = seq_6d.view(seq_6d.shape[0], -1)

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

class H36MSeq2Seq(Dataset):
    def __init__(self,
                 data_dir      : str,
                 split         : str  = 'train',
                 max_seq_len   : int  = 75,
                 sixd_scale    : float = 1.0,
                 hist_len= 50,
                 pred_len= 25,):
        """
        Parameters
        ----------
        data_dir : str        存放 h36m_xxx_xxx.npz 的目录
        split    : str        'train' / 'val' / 'test'
        max_seq_len : int     把长序列切成这个长度的 clip
        sixd_scale : float    对 6D 旋转做缩放（如 1.0 表示不变）
        """
        super().__init__()
        self.max_seq_len = max_seq_len
        self.sixd_scale  = sixd_scale
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.split= split

        # ---------- 1. 读取 npz ----------
        npz_path = os.path.join(data_dir, f'h36m_{split}.npz') \
                   if not os.path.exists(os.path.join(
                       data_dir, f'h36m_{split}_{max_seq_len}.npz')) \
                   else os.path.join(data_dir, f'h36m_{split}_{max_seq_len}.npz')

        data = np.load(npz_path, allow_pickle=True)
        all_sixd  = data['sampled_sixd_seq']   # (N,T,V,6)
        all_xyz   = data['sampled_xyz_seq']    # (N,T,V,3)
        all_label = data['label_seq']          # (N,)
        data.close()

        # ---------- 2. 拆分为 clips ----------
        self.processed_sixd   = []   # list[n_clip]<numpy(Tc,V,6)>
        self.processed_xyz    = []   # list[n_clip]<numpy(Tc,V,3)>
        self.processed_label  = []   # list[int]
        for seq_6d, seq_xyz, seq_label in tqdm(
                zip(all_sixd, all_xyz, all_label),
                total=len(all_sixd), desc=f'Processing {split}'):
            seq_6d = seq_6d * self.sixd_scale
            clips_6d  = self._split_seq(seq_6d,  self.max_seq_len)
            clips_xyz = self._split_seq(seq_xyz, self.max_seq_len)
            self.processed_sixd.extend(clips_6d)
            self.processed_xyz.extend(clips_xyz)
            self.processed_label += [int(seq_label)] * len(clips_6d)

        # ---------- 3. 转为 Tensor ----------
        self.processed_sixd  = [c.float() for c in self.processed_sixd]
        self.processed_xyz   = [c.float() for c in self.processed_xyz]
        self.processed_label = torch.tensor(self.processed_label, dtype=torch.long)

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.processed_sixd)

    def __getitem__(self, idx):
        """
        返回：
            - sixd_clip : Tensor (T, V, 6)
            - xyz_clip  : Tensor (T, V, 3)
            - label     : int
        """
        if self.split == 'train' or self.split == 'val':
            return (self.processed_sixd[idx][:self.hist_len,...],     # (T,V,6)
                    self.processed_sixd[idx][self.hist_len:,...],      # (T,V,3)
                    self.processed_label[idx])    # int
        elif self.split == 'test':
            return (self.processed_sixd[idx][:self.hist_len,...],     # (T,V,6)
                    self.processed_sixd[idx][self.hist_len:,...],      # (T,V,3)
                    self.processed_xyz[idx][self.hist_len:,...],      # (T,V,3)
                    self.processed_label[idx])


    # ------------------------------------------------------------
    @staticmethod
    def _split_seq(seq: np.ndarray, max_len: int):
        """
        将 numpy 序列按 max_len 切分：
            输入  shape (T, V, C)
            输出  List[np.ndarray]  len = floor(T / max_len)
        """
        n_clip = seq.shape[0] // max_len
        if n_clip == 0:
            return []        # 丢弃过短序列
        seq = seq[: n_clip * max_len]
        return np.split(seq, n_clip, axis=0)

class H36MClfDataset(Dataset):
    def __init__(self,
                 data_dir      : str,
                 split         : str  = 'train',
    ):

        super().__init__()

        self.split= split

        # ---------- 1. 读取 npz ----------
        npz_path = os.path.join(data_dir, f'h36m_{split}.npz') \
                   if not os.path.exists(os.path.join(
                       data_dir, f'h36m_{split}.npz')) \
                   else os.path.join(data_dir, f'h36m_{split}.npz')

        data = np.load(npz_path, allow_pickle=True)
        all_sixd  = data['sampled_sixd_seq']   # (N,T,V,6)
        all_label = data['label_seq']          # (N,)
        data.close()

        self.processed_sixd   = []   # list[n_clip]<numpy(Tc,V,6)>
        self.processed_xyz    = []   # list[n_clip]<numpy(Tc,V,3)>
        self.processed_label  = []   # list[int]


        for seq_6d, seq_label in tqdm(
                zip(all_sixd, all_label),
                total=len(all_sixd), desc=f'Processing {split}'):
            # 下采样，采样频率下降一半
            seq_6d = seq_6d[::4, ...]  # (T, V, 6)
            self.processed_sixd.append(seq_6d)
            self.processed_label.append(seq_label)

        # ---------- 3. 转为 Tensor ----------
        self.processed_sixd  = [c.float() for c in self.processed_sixd]
        self.processed_label = torch.tensor(np.array(self.processed_label), dtype=torch.long)
    # ------------------------------------------------------------
    def __len__(self):
        return len(self.processed_sixd)


    def __getitem__(self, idx):
        return (self.processed_sixd[idx],     # (T,V,6)
                self.processed_label[idx])    # int

def collate_motion(batch, pad_value=0.0):
    """
    batch: List[Tuple[Tensor(seq_len, V, 6),  label(int)]]
    返回:
        padded_x : (B, L_max, V, 6)
        lengths  : (B,)   原始长度
        y        : (B,)
    """
    seqs, labels = zip(*batch)           # 拆 list
    lengths = torch.tensor([s.size(0) for s in seqs], dtype=torch.long)
    # pad_sequence 会在第 0 维补齐 → (L_max, B, V, 6)
    padded = pad_sequence(seqs, batch_first=True, padding_value=pad_value)
    # pad_sequence 默认把序列维度当第 0 维，这里设 batch_first=True 得到 (B,L_max,V,6)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels

def create_h36m_clf_dataloader(
        dataset,
        batch_size     : int,
        pin_memory    : bool = True,
        shuffle       : bool = True,
        drop_last     : bool = False,
        collate_fn    : Callable = collate_motion,
):
    """
    创建 H36M 分类数据集的 dataloader
    :param data_dir:
    :param batch_size:
    :param split:
    :param num_workers:
    :param pin_memory:
    :param shuffle:
    :param drop_last:
    :param collate_fn:
    :return:
    """
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn
    )

    return dataloader

class SyllableDataset(Dataset):
    """Dataset for motion syllable evaluation/visualisation."""

    def __init__(
        self,
        data_dir: str,
        split: str = "val",
        downsample_rate: int = 2,
    ) -> None:
        super().__init__()

        npz_path = os.path.join(data_dir, f"h36m_{split}.npz")
        if not os.path.isfile(npz_path):
            raise FileNotFoundError(f"NPZ file not found: {npz_path}")

        data = np.load(npz_path, allow_pickle=True)
        self.xyz_all = data["sampled_xyz_seq"]        # (N,T,V,3)
        self.rot6d_all = data.get("sampled_sixd_seq", None)  # (N,T,V,6) optional
        self.labels = data.get("label_seq", np.zeros(len(self.xyz_all), dtype=int))
        data.close()

        self.xyz, self.rot6d, self.action = [], [], []
        for xyz, rot6d, lbl in tqdm(
            zip(self.xyz_all, self.rot6d_all, self.labels),
            total=len(self.xyz_all), desc=f"Down‑sampling {split}"):
            xyz_ds = xyz[::downsample_rate]  # (T',V,3)
            rot_ds = (
                rot6d[::downsample_rate]
                if rot6d is not None else np.zeros((*xyz_ds.shape[:2], 6), np.float32)
            )
            self.xyz.append(xyz_ds)
            self.rot6d.append(rot_ds)
            self.action.append(int(lbl))

    # ------------------------------------------------------------
    def __len__(self):
        return len(self.xyz)

    def __getitem__(self, idx):
        return self.rot6d[idx], self.xyz[idx], self.action[idx]

def collate_fn(batch, pad_val=0.0):
    rot6d, xyz, lbl = zip(*batch)
    len_t = torch.tensor([r.size(0) for r in rot6d], dtype=torch.long)
    rot_pad = pad_sequence(rot6d, batch_first=True, padding_value=pad_val)  # (B,L,V,6)
    xyz_pad = pad_sequence(xyz, batch_first=True, padding_value=pad_val)    # (B,L,V,3)
    lbl = torch.tensor(lbl, dtype=torch.long)
    return rot_pad, xyz_pad, len_t, lbl
if __name__=="__main__":


    train_set = H36MClfDataset('../data/h3.6m', 'train',
                           )
    dataloader = create_h36m_clf_dataloader(train_set, batch_size=32, shuffle=True)

    # 验证数据形状
    x, l,y = next(iter(dataloader))
    print(f"输入数据形状: {x.shape}")  # 期望形状 (32, T, V*C)
    print(f"输出数据形状: {y.shape}")  # 期望形状 (32, T, V*C)
    x, l,y = next(iter(dataloader))
    print(f"输入数据形状: {x.shape}")  # 期望形状 (32, T, V*C)
    print(f"输出数据形状: {y.shape}")  # 期望形状 (32, T, V*C)

    train_set = H36MSeq2Seq('../data/h3.6m','train',
                            75, 1.0)
    dataloader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)

    # 验证数据形状
    hist,future ,_= next(iter(dataloader))
    print(f"输入数据形状: {hist.shape}")  # 期望形状 (32, T, V*C)
    print(f"输出数据形状: {future.shape}")  # 期望形状 (32, T, V*C)

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
    # dataset = HumanVQVAESixDDataSet(data_dir='../data/h3.6m', split='train')
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    #
    # # 验证数据形状
    # sample,_ ,_= next(iter(dataloader))
    # print(f"输入数据形状: {sample.shape}")  # 期望形状 (32, T, V*C)
    #
    # # 测试VQ-VAE重构
    # from model.vq_vae import HumanVQVAE
    # nb_code = 128
    # code_dim = 128
    # output_emb_width = 128
    # down_t = 3
    # stride_t = 2
    # width = 1024
    # depth = 3
    # dilation_growth_rate = 3
    # activation = 'relu'
    # norm = None
    #
    #
    # class Args:
    #     def __init__(self, dataname, quantizer):
    #         self.dataname = dataname
    #         self.quantizer = quantizer
    #         self.commit = 0.02  # 量化器相关参数
    #         self.gamma = 0.99  # EMA衰减系数
    #         self.mu = 0.99  # 重置量化器参数（若需要）
    #
    #
    # model = HumanVQVAE(
    #     Args('h36m','ema_reset'),
    #     nb_code=nb_code,
    #     code_dim=code_dim,
    #     output_emb_width=output_emb_width,
    #     down_t=down_t,
    #     stride_t=stride_t,
    #     width=width,
    #     depth=depth,
    #     dilation_growth_rate=dilation_growth_rate,
    #     activation=activation,
    #     norm=norm
    # )  # 需要定义args参数
    # reconstructed, _, _ = model(sample)
    # print(f"重构输出形状: {reconstructed.shape}")  # 应与输入形状一致
    # from thop import profile, clever_format
    # flops, params = profile(model, inputs=(sample,))
    #
    # print('FLOPs:', flops / 1e6, 'M')  # 将FLOPs转换为G（十亿）
    # print('Params:', params / 1e6, 'M')  # 将参数量转换为M（百万）