#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# eval_vq_config.py (改名以反映其使用配置文件)

import argparse
import time
from datetime import datetime
import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from types import SimpleNamespace
import matplotlib.pyplot as plt

from model.motion_vqvae import MotionVQVAE  # 假设主要使用这个
# from model.transformer_vqvae import TransformerVQVAE # 如果要用Transformer VQVAE，取消注释
from utils.forward_kinematics import sixd_to_xyz_torch  # 确保此路径正确
from utils.func import import_class  # 需要这个函数
from utils.data_utils import get_adjacency_matrix, get_motion_chains  # 确保此路径正确
from pathlib import Path
from utils.forward_kinematics import sixd_to_xyz_babel,sixd_to_xyz_3dpw
from utils.metrics import compute_acceleration_error, compute_pa_mpjpe




def _dotdict(d: dict):
    """dict -> argparse.Namespace"""
    return SimpleNamespace(**d)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    # 合并默认参数（如果需要）
    # default_args = get_default_eval_args() # 你可以创建一个函数返回默认参数
    # cfg_dict = {**vars(default_args), **cfg_dict}
    return _dotdict(cfg_dict)


class EvaluatorVQ:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)

        # 创建工作目录
        if args.work_dir:
            # 用配置中的 exp_name (如果有的话) 或者模型路径的一部分来命名评估目录，避免仅用时间戳
            model_name_part = args.exp_name
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.work_dir = Path(args.work_dir) / f"{args.dataset}_{model_name_part}_{timestamp}"
        else:
            timestamp= datetime.now().strftime('%Y%m%d_%H%M%S')
            self.work_dir = Path("./eval_output_default") / f"{args.dataset}_{timestamp}"

        self.work_dir.mkdir(parents=True, exist_ok=True)

        if getattr(args, 'save_visualizations', False):
            self.vis_dir = self.work_dir / "visualizations"
            self.vis_dir.mkdir(parents=True, exist_ok=True)

        self.all_mpjpe_errors = []  # 存储每个序列的平均MPJPE (T上平均, V上平均)
        self.all_6d_errors = []  # 存储每个序列的平均6D L2误差 (T上平均, V上平均)
        self.all_pa_mpjpe_errors = []
        self.all_acc_errors = []


        # 用于可视化每个样本的关节误差传播
        self.per_sample_joint_mpjpe = []  # list of (V,) arrays
        self.per_sample_joint_6derr = []  # list of (V,) arrays

        # 1. 加载模型
        self.net = self.build_model()
        # 2. 加载数据
        self.test_loader = self.build_dataloader()
        # 3. 获取运动链
        self.motion_chains = get_motion_chains(args.dataset)

        if not hasattr(self.args, 'smplh_model_base_path'):
            self.args.smplh_model_base_path = "./amass/smplh"  # Default, should be in config
            self.print_log(
                f"Warning: 'smplh_model_base_path' not in config, using default: {self.args.smplh_model_base_path}")
            # Add num_betas_model for BodyModel to args if not present
        if not hasattr(self.args, 'num_betas_model'):
            self.args.num_betas_model = 16  # Default, should be in config
            self.print_log(
                f"Warning: 'num_betas_model' (for BodyModel) not in config, using default: {self.args.num_betas_model}")


    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.args.print_log:
            with open(self.work_dir / 'log.txt', 'a') as f:
                print(s, file=f)

    def build_model(self):
        # 确保这些参数与配置文件中的模型参数部分匹配
        net = MotionVQVAE(
            dataset_name=self.args.dataset,  # 传递数据集名称给模型
            n_heads=self.args.n_heads,
            num_joints=self.args.nb_joints,  # nb_joints from config
            in_dim=self.args.in_dim,
            n_codebook=self.args.n_codebook,
            balance=self.args.balance,
            n_e=self.args.n_e,
            e_dim=self.args.e_dim,
            hid_dim=self.args.hid_dim,
            beta=self.args.beta,
            quan_min_pro=self.args.quant_min_prop,  # Renamed in train script
            n_layers=self.args.n_layers,
            seq_len=self.args.seq_len,  # or args.window_size
        ).to(self.device)

        self.print_log(f"Loading model from {self.args.eval_model_path}")
        ckpt = torch.load(self.args.eval_model_path, map_location=self.device)
        if 'loss' in ckpt:
            self.print_log(f"Checkpoint's best validation loss: {ckpt['loss']:.6f}")
        net.load_state_dict(ckpt['net'])
        net.eval()
        return net

    def build_dataloader(self):
        DatasetClass = import_class(self.args.dataset_class)
        # 为数据集类准备参数
        dataset_args = {
            "data_dir": self.args.data_root,
            "split": 'test',  # 强制为test
            "max_seq_len": self.args.window_size,  # 使用window_size作为序列长度
        }
        # 为特定数据集添加额外参数
        if "Babel" or "3dpw" in self.args.dataset_class:
            dataset_args["num_betas"] = self.args.num_betas_model
            # dataset_args["num_body_joints"] 已由模型中的 nb_joints 间接定义

        dataset = DatasetClass(**dataset_args)
        loader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False)
        return loader

    def evaluate(self):
        with torch.no_grad():
            for batch_idx, data_batch in enumerate(tqdm(self.test_loader, desc="Evaluating")):


                batch_6d_input_model = None # This will be (B, T, V_model, 6)
                batch_smpl_betas = None   # (B, num_betas_data)
                batch_smpl_trans = None   # (B, T, 3)
                batch_smpl_genders = None # List of B strings


                if self.args.dataset in ['babel','3dpw']:
                    if not isinstance(data_batch, dict) or 'seq_6d' not in data_batch:
                        self.print_log(f"Error: Expected dict with 'seq_6d' for BABEL test, got {type(data_batch)}. Skipping batch.")
                        continue
                    batch_6d_input_model = data_batch['seq_6d'].to(self.device).float()
                    batch_smpl_betas = data_batch['betas'].to(self.device).float()
                    batch_smpl_trans = data_batch['trans'].to(self.device).float()
                    batch_smpl_genders = data_batch['gender'] # List of strings, no .to(device)
                else: # Assuming H3.6M format (tuple: 6d, xyz, label)
                    if len(data_batch) < 2:
                        self.print_log(f"Error: Expected at least 2 items in data_batch for {self.args.dataset}, got {len(data_batch)}. Skipping batch.")
                        continue
                    batch_6d_input_model = data_batch[0].to(self.device).float()

                B, T, V_model, C_6d = batch_6d_input_model.shape
                (recon_6d,), _, _ = self.net(x=batch_6d_input_model, y=0,
                                             valid=torch.ones(B, T).to(self.device))

                # --- Calculate 6D Error ---
                sixd_diff = recon_6d - batch_6d_input_model
                sixd_err_per_joint_per_frame = torch.norm(sixd_diff, p=2, dim=-1)
                mean_6d_err_per_sample = sixd_err_per_joint_per_frame.mean(dim=[1, 2])  # Avg over T and V_model
                self.all_6d_errors.extend(mean_6d_err_per_sample.cpu().tolist())


                # --- Convert to 3D XYZ and Calculate MPJPE ---
                pred_xyz = None
                batch_xyz_gt = None
                if self.args.dataset =='babel':
                    pred_xyz = sixd_to_xyz_babel(
                        recon_6d, batch_smpl_betas, batch_smpl_trans, batch_smpl_genders,
                        self.args.smplh_model_base_path, self.args.num_betas_model, self.device
                    )
                    batch_xyz_gt = sixd_to_xyz_babel(  # GT XYZ from GT 6D poses
                        batch_6d_input_model, batch_smpl_betas, batch_smpl_trans, batch_smpl_genders,
                        self.args.smplh_model_base_path, self.args.num_betas_model, self.device
                    )
                elif self.args.dataset == '3dpw':
                    pred_xyz = sixd_to_xyz_3dpw(
                        recon_6d, batch_smpl_betas, batch_smpl_trans, batch_smpl_genders,
                        self.args.smplh_model_base_path, self.device
                    )
                    batch_xyz_gt = sixd_to_xyz_3dpw(  # GT XYZ from GT 6D poses
                        batch_6d_input_model, batch_smpl_betas, batch_smpl_trans, batch_smpl_genders,
                        self.args.smplh_model_base_path,  self.device
                    )
                elif self.args.dataset =='h36m':
                    # Use the generic or dataset-specific sixd_to_xyz_torch for these
                    # Assuming sixd_to_xyz_torch is defined elsewhere or needs to be adapted
                    # For now, let's use a placeholder as it was in original eval script
                    try:
                        # You'll need to import/define your original sixd_to_xyz_torch
                        from utils.forward_kinematics import sixd_to_xyz_torch  # Example path
                        pred_xyz = sixd_to_xyz_torch(recon_6d)
                        batch_xyz_gt = data_batch[1].to(self.device).float()  # Directly from dataloader
                    except ImportError:
                        self.print_log("Warning: 'sixd_to_xyz_torch' not found. MPJPE for H3.6M/3DPW will be skipped.")
                        pred_xyz = torch.zeros_like(batch_6d_input_model[..., :3])  # Placeholder
                        batch_xyz_gt = data_batch[1].to(self.device).float() if len(data_batch) > 1 and data_batch[
                            1] is not None else pred_xyz

                else:
                    self.print_log(
                        f"Warning: XYZ conversion not implemented for dataset {self.args.dataset}. MPJPE will be skipped.")
                    # Assign placeholders to avoid crashing if MPJPE calculation is attempted
                    pred_xyz = torch.zeros_like(batch_6d_input_model[..., :3])
                    batch_xyz_gt = torch.zeros_like(batch_6d_input_model[..., :3])
                    # MPJPE Calculation (ensure pred_xyz and batch_xyz_gt are valid)
                if pred_xyz is not None and batch_xyz_gt is not None and \
                        pred_xyz.shape == batch_xyz_gt.shape:  # Basic shape check

                    # The BodyModel for BABEL outputs full SMPL-H joints (e.g., 52).
                    # The VQVAE model (nb_joints) might be trained on fewer (e.g., 22 for BABEL root+body).
                    # For a fair MPJPE on BABEL, we should compare on the common set of joints
                    # that the VQVAE was *intended* to represent, if an explicit mapping exists,
                    # OR compare on all output joints from BodyModel if GT is also full BodyModel output.
                    # Current sixd_to_xyz_babel returns full BodyModel joints.
                    # So, comparison is on these full joints.
                    mpjpe_diff = pred_xyz - batch_xyz_gt
                    mpjpe_per_joint_per_frame = torch.norm(mpjpe_diff, p=2, dim=-1)  # (B, T, V_output_smpl)
                    mean_mpjpe_per_sample = mpjpe_per_joint_per_frame.mean(
                        dim=[1, 2])  # Avg over T and V_output_smpl
                    self.all_mpjpe_errors.extend(mean_mpjpe_per_sample.cpu().tolist())
                    # ---- 计算 PA-MPJPE -----
                    pa_mpjpe_vals = compute_pa_mpjpe(pred_xyz, batch_xyz_gt)  # (B,)
                    self.all_pa_mpjpe_errors.extend(pa_mpjpe_vals.cpu().tolist())

                    # ---- 计算 Acceleration Error -----
                    acc_vals = compute_acceleration_error(pred_xyz, batch_xyz_gt)  # (B,)
                    self.all_acc_errors.extend(acc_vals.cpu().tolist())
                    # For visualization data collection
                    if getattr(self.args, 'save_visualizations', False) and \
                            len(self.per_sample_joint_mpjpe) < getattr(self.args, 'num_samples_to_visualize', 0):

                        # Time average: (B, V_model) for 6D, (B, V_output_smpl) for 3D
                        avg_6derr_per_joint_b = sixd_err_per_joint_per_frame.mean(dim=1).cpu().numpy()
                        avg_mpjpe_per_joint_b = mpjpe_per_joint_per_frame.mean(dim=1).cpu().numpy()

                        for k_sample in range(B):
                            if len(self.per_sample_joint_mpjpe) < getattr(self.args, 'num_samples_to_visualize', 0):
                                self.per_sample_joint_6derr.append(avg_6derr_per_joint_b[k_sample])
                                self.per_sample_joint_mpjpe.append(avg_mpjpe_per_joint_b[k_sample])
                            else:
                                break
                else:
                    self.print_log(
                        f"Skipping MPJPE calculation for batch {batch_idx} due to missing/mismatched XYZ data.")


        self.summarize_and_visualize_results()

    def summarize_and_visualize_results(self):
        overall_avg_mpjpe = np.mean(self.all_mpjpe_errors) if self.all_mpjpe_errors else float('nan')
        overall_avg_6derr = np.mean(self.all_6d_errors) if self.all_6d_errors else float('nan')
        overall_avg_pa_mpjpe = np.mean(self.all_pa_mpjpe_errors) if self.all_pa_mpjpe_errors else float('nan')
        overall_avg_acc = np.mean(self.all_acc_errors) if self.all_acc_errors else float('nan')


        if self.args.dataset in ['babel','3dpw']:
            overall_avg_mpjpe *= 100
            overall_avg_pa_mpjpe *= 100  # m → mm
            overall_avg_acc *= 100


        self.print_log(f"\n--- Evaluation Summary ---")
        self.print_log(f"Dataset: {self.args.dataset}")
        self.print_log(f"Overall Average MPJPE      : {overall_avg_mpjpe:.4f} mm")
        self.print_log(f"Overall Average PA-MPJPE   : {overall_avg_pa_mpjpe:.4f} mm (Procrustes aligned)")
        self.print_log(f"Overall Average 6D L2 Error: {overall_avg_6derr:.4f}")
        self.print_log(f"Overall Average Accel Err  : {overall_avg_acc:.4f} mm/frame^2")
        # 可视化误差传播
        if getattr(self.args, 'save_visualizations', False) and self.motion_chains:
            self.print_log(f"Saving error propagation visualizations to: {self.vis_dir}")
            num_to_plot = min(len(self.per_sample_joint_mpjpe), getattr(self.args, 'num_samples_to_visualize', 0))

            for i in range(num_to_plot):
                mean6d_per_joint = self.per_sample_joint_6derr[i]  # (V_model,)
                mean3d_per_joint = self.per_sample_joint_mpjpe[i]  # (V_aligned,)

                # Ensure V_model and V_aligned are handled correctly for indexing motion_chains
                # Assume motion_chains uses indices compatible with the smaller of V_model, V_aligned if they differ.
                # Or, ideally, V_model from VQVAE output should match the V for which motion_chains are defined.
                # And V_aligned (from MPJPE calc) should also match.

                num_joints_for_chains = self.args.nb_joints  # Use the model's number of joints for chain indexing

                for chain_idx, chain_joint_indices in enumerate(self.motion_chains):
                    # Filter chain_joint_indices to be within the bounds of available joint errors
                    valid_chain_indices = [idx for idx in chain_joint_indices if
                                           idx < num_joints_for_chains and idx < len(mean6d_per_joint) and idx < len(
                                               mean3d_per_joint)]
                    if not valid_chain_indices:
                        continue

                    chain_err6d_vals = mean6d_per_joint[valid_chain_indices]
                    chain_err3d_vals = mean3d_per_joint[valid_chain_indices]

                    plt.figure(figsize=(10, 6))
                    x_labels = [str(j_idx) for j_idx in valid_chain_indices]
                    x_pos = np.arange(len(x_labels))

                    plt.plot(x_pos, chain_err6d_vals, marker='o', linestyle='-',
                             label=f'Avg 6D Error (x{self.args.sixd_scale:.1f})')
                    plt.plot(x_pos, chain_err3d_vals * 1000, marker='s', linestyle='-',
                             label='Avg MPJPE (mm)')  # MPJPE often in mm

                    plt.xticks(x_pos, x_labels)
                    plt.xlabel(f'Joint ID along chain (Dataset: {self.args.dataset})')
                    plt.ylabel('Mean Error')
                    plt.title(f'Sample {i}, Chain {chain_idx} - Avg Error Propagation')
                    plt.legend()
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig(self.vis_dir / f'sample_{i}_chain_{chain_idx}_avg_error.png')
                    plt.close()
            self.print_log(f"Finished saving visualizations for {num_to_plot} samples.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Motion-VQVAE from config")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML evaluation config file")
    cli_args = parser.parse_args()
    args = load_config(cli_args.config)  #

    evaluator = EvaluatorVQ(args)  #
    evaluator.evaluate()  #


if __name__ == "__main__":
    main()