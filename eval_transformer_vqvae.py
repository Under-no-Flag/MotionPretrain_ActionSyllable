# eval_vq_label.py

import argparse
import time
from datetime import datetime

import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

# VQVAE 模型
from model.motion_vqvae import MotionVQVAE
from model.transformer_vqvae import TransformerVQVAE
# 修改后的数据集
from dataset.h36m import HumanVQVAESixDDataSet
# 将 6D 转成 3D
from options import option_vq
from utils.forward_kinematics import sixd_to_xyz_torch
from utils.func import str2bool
from matplotlib import pyplot as plt

def mpjpe_per_frame(pred_xyz, gt_xyz):
    """
    计算 MPJPE:
    pred_xyz, gt_xyz 均为 (B, T, V, 3).
    返回形状 (B, T) 或 (B,) 均可, 看需求.

    这里示例按 (B, T) 返回，先对 V 做平均。
    """
    # (B, T, V, 3)
    diff = pred_xyz - gt_xyz
    dist = torch.norm(diff, dim=-1)   # (B, T, V)
    mpjpe = dist.mean(dim=-1)        # 在 V 上做平均，得到 (B, T)
    return mpjpe



class EvaluatorVQ:
    def __init__(self, args):
        if args.work_dir:
            args.work_dir=args.work_dir+datetime.now().strftime('%Y%m%d_%H%M%S')
        self.args = args
        self.device = torch.device(args.device)
        if not os.path.exists(self.args.work_dir):
            os.makedirs(self.args.work_dir)
        # 动作类别的名称（示例）
        self.actions = [
            'posing','greeting','sitting','walking','smoking','walkingtogether',
            'phoning','walkingdog','waiting','eating','discussion','purchases',
            'sittingdown','directions','takingphoto','misc'
        ]
        # 按类别记录 MPJPE
        # 每个类别对应一个列表，存储所有样本(帧)的误差
        self.test_mpjpe = {i: [] for i in range(self.args.num_class)}
        self.test_6derr = {i: [] for i in range(self.args.num_class)}

        # 1. 加载模型
        self.net = self.build_model()
        # 2. 加载数据
        self.test_loader = self.build_dataloader()
        self.motion_chain = [
            [0, 1, 2, 3, 4, 5],
            [0, 6, 7, 8, 9, 10],
            [0, 11, 12, 13, 14, 15],
            [12, 16, 17, 18, 19, 20, 21, 22, 23],
            [12, 24, 25, 26, 27, 28, 29, 30, 31],
        ]

    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.args.print_log:
            with open(os.path.join(self.args.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

    def build_model(self):
        # (要与训练时的配置保持一致)
        class DummyArgs:
            def __init__(self):
                self.dataname = 'h36m'
                self.quantizer = 'ema_reset'
                self.commit = 0.02
                self.gamma = 0.99
                self.mu = 0.99

        dummy_args = DummyArgs()
        # net = TransformerVQVAE(
        #     n_heads=4,
        #     in_dim=6,
        #     num_joints=32,
        #     n_codebook=32,
        #     balance=0,
        #     n_e=512,
        #     e_dim=256,
        #     hid_dim=16,
        #     beta=0.25,
        #     quant_min_prop=1.0,
        #     n_layers=[0, 16],
        #     seq_len=64,
        #     causal_encoder=True,
        # )

        net = MotionVQVAE(
            n_heads=4,
            num_joints=32,
            in_dim=6,
            n_codebook=6,
            balance=0,
            n_e=1024,
            e_dim=128,
            hid_dim=128,
            beta=0.25,
            quant_min_prop=1.0,
            n_layers=[0, 10],
            seq_len=64,
        )
        print("Loading model from", self.args.model_path)
        net.to(self.device)
        # 加载权重
        ckpt = torch.load(self.args.model_path, map_location=self.device)
        print("best val recon loss",ckpt['loss'])
        net.load_state_dict(ckpt['net'])
        net.eval()
        return net

    def build_dataloader(self):
        dataset = HumanVQVAESixDDataSet(
            data_dir=self.args.data_dir,
            split='test',
            input_length=self.args.input_length,
            predicted_length=self.args.predicted_length,
            max_seq_len=self.args.max_seq_len,
            sixd_scale=self.args.sixd_scale
        )
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        return loader

    def evaluate(self):
        with torch.no_grad():
            for batch_idx,(batch_6d, batch_xyz, batch_label) in enumerate(self.test_loader):
                # batch_6d:   (B, T, V*6)  (示例)
                # batch_xyz:  (B, T, V, 3)
                # batch_label:(B, )

                batch_6d   = batch_6d.to(self.device)
                batch_xyz  = batch_xyz.to(self.device)
                batch_label= batch_label.to(self.device)

                # 前向推理，得到重构的 6D
                (recon_6d,), quant_item, _ = self.net(x=batch_6d, y=0,
                                                  valid=torch.ones(batch_6d.shape[0], batch_6d.shape[1]).to(
                                                      batch_6d.device))


                # (1) --- 计算 6D 重构误差 ---
                sixd_diff = recon_6d - batch_6d  # (B, T, V,6)
                # 对最后一维 (V*6) 做 L2 norm => (B, T)
                # 注意：torch.norm(..., dim=-1) 是 sqrt(∑x^2)，如果想要 mean square 可以改成 squared, etc.
                sixd_err_j = torch.norm(sixd_diff, dim=-1)  # (B, T, V)

                sixd_l2 = torch.norm(sixd_diff, dim=-1).mean(axis=-1)  # (B, T)

                # 调整形状: (B, T, V*6) -> (B, T, V, 6)
                B, T, V,C = recon_6d.shape
                # V = VC // 6
                # recon_6d = recon_6d.view(B, T, V, 6)


                # 转 3D
                # 如果 sixd_to_xyz_torch 要求 (B, T, V, 6)，则可以直接传
                pred_xyz = sixd_to_xyz_torch(recon_6d)  # (B, T, V, 3)

                diff = pred_xyz - batch_xyz  # (B, T, V, 3)
                mpjpe_j = torch.norm(diff, dim=-1)  # (B, T, V)
                dist = torch.norm(diff, dim=-1)  # (B, T, V)
                mpjpe_b_t = dist.mean(dim=-1)  # (B, T)

                # 按动作类别，存入 self.test_mpjpe[label]
                # 假设 self.test_mpjpe[action_label] 是一个 list，用来装形状 (T,) 的张量
                for i in range(batch_6d.shape[0]):
                    a_label = int(batch_label[i].item())
                    # 取出该样本的 (T,) mpjpe
                    sample_error = mpjpe_b_t[i]  # shape: (T,)
                    # 转到 CPU，避免后续堆叠时 GPU/CPU 混合
                    sample_error = sample_error.detach().cpu().numpy()
                    self.test_mpjpe[a_label].append(sample_error)

                    # 6D error
                    sample_error_6d = sixd_l2[i].detach().cpu().numpy()  # shape = (T,)
                    self.test_6derr[a_label].append(sample_error_6d)

                # for i in range(B):
                #     # 转到 numpy 方便处理
                #     err6d_np = sixd_err_j[i].detach().cpu().numpy()  # (T, V)
                #     err3d_np = mpjpe_j[i].detach().cpu().numpy()  # (T, V)
                #
                #     # 时序上取平均，得到每个关节的平均误差
                #     mean6d_per_joint = err6d_np.mean(axis=0)  # (V,)
                #     mean3d_per_joint = err3d_np.mean(axis=0)  # (V,)
                #
                #     # 对每条运动链，抽取对应关节的误差并打印／保存
                #     for chain_idx, chain in enumerate(self.motion_chain):
                #         chain_ids = np.array(chain)
                #         chain_err6d = mean6d_per_joint[chain_ids]  # (chain_len,)
                #         chain_err3d = mean3d_per_joint[chain_ids]
                #
                #         # —— 打印表格 ——
                #         header = "Joint".ljust(8) + "".join(f"{jid:>8}" for jid in chain_ids)
                #         row6d = "6D_err".ljust(8) + "".join(f"{v:8.3f}" for v in chain_err6d)
                #         row3d = "MPJPE".ljust(8) + "".join(f"{v:8.3f}" for v in chain_err3d)
                #         print(f"\nSample {i}, Chain {chain_idx}:\n{header}\n{row6d}\n{row3d}\n")
                #
                #         x = np.arange(len(chain_ids))
                #         plt.figure()
                #         plt.plot(x, chain_err6d*1000, marker='o', label='6D_error')
                #         plt.plot(x, chain_err3d, marker='s', label='MPJPE')
                #         plt.xticks(x, chain_ids)
                #         plt.xlabel('Joint ID along chain')
                #         plt.ylabel('Mean Error')
                #         plt.title(f'Sample {i} Chain {chain_idx} Error Propagation')
                #         plt.legend()
                #         plt.savefig(f'./vis_chain_error/batch_{batch_idx}chain_error_sample_{i}_chain_{chain_idx}.png')
        # 全部评估结束后，打印结果
        self.summarize_results()

    def summarize_results(self):
        # 假设和 MPJPE 部分一致，T 为预测长度 (比如 25)
        T = len(self.test_mpjpe[0][0])  # 例如从第 0 类取一个样本看下长度

        # --- 先打印 MPJPE（你已有的代码） ---
        self.print_log("=== Per-Class MPJPE ===")
        self._print_action_error_table(self.test_mpjpe, T, title="MPJPE")

        # --- 再打印 6D 重构误差 ---
        self.print_log("=== Per-Class 6D Error ===")
        self._print_action_error_table(self.test_6derr, T, title="6D Error")

    def _print_action_error_table(self, error_dict, T, title=""):
        """
        与 MPJPE 打印逻辑类似：
        error_dict[action_idx] 中放的都是若干 shape=(T,) 的numpy array
        """
        pred_time_idx = np.arange(T)
        # 表头
        print_str = "{0: <16} |".format("milliseconds")
        for ms in (pred_time_idx + 1) * 40:
            print_str += f" {ms:5d} |"
        self.print_log(print_str)

        # (num_actions, T)
        avg_error_ms = np.zeros((len(self.actions), T))

        for action_num, action_name in enumerate(self.actions):
            if len(error_dict[action_num]) == 0:
                continue
            # 堆叠: (N_samples, T)
            arr = np.stack(error_dict[action_num], axis=0)
            # 对样本维度做平均 -> (T,)
            mean_err_t = arr.mean(axis=0)

            line_str = "{0: <16} |".format(action_name)
            for ms_idx in range(T):
                val = mean_err_t[ms_idx]
                avg_error_ms[action_num, ms_idx] = val
                line_str += f" {val:.3f} |"
            self.print_log(line_str)

        # 打印一行 "Average"
        avg_str = "{0: <16} |".format(f"Average {title}")
        for ms_idx in range(T):
            valid_vals = []
            for action_num in range(len(self.actions)):
                if len(error_dict[action_num]) > 0:
                    valid_vals.append(avg_error_ms[action_num, ms_idx])
            if len(valid_vals) == 0:
                avg_str += "  N/A  |"
            else:
                avg_str += f" {np.mean(valid_vals):.3f} |"
        self.print_log(avg_str)


def main():

    args =option_vq.get_args_parser()
    evaluator = EvaluatorVQ(args)
    evaluator.evaluate()

if __name__ == "__main__":
    main()
