# eval.py
import argparse
import time
from datetime import datetime

import yaml
import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset.h36m import Human36mDataset
from model.MotionGPT import MotionGPT
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.forward_kinematics import sixd_to_xyz_torch

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_parser():
    parser = argparse.ArgumentParser(description='MotionGPT Evaluation')
    parser.add_argument('--config', type=str, default=None, help='Path to config file')
    parser.add_argument('--data_dir', type=str, default='./data/h3.6m', help='Path to dataset')
    parser.add_argument('--model_path', type=str, default='./checkpoints/best_model.pth', help='Path to trained model')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--save_dir', type=str, default='results/', help='Path to save visualizations')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')


    parser.add_argument(
        '--num_class',
        type=int,
        default=16,
        help='Debug mode; default false')

    parser.add_argument(
        '--xyz_scale',
        type=float,
        default=100.,
    )

    parser.add_argument(
        '--work-dir',
        type=str,
        default='./work_dir/h36m',
        help='the work folder for storing results')

    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')


    return parser


def visualize_predictions(inputs, preds, targets, save_path):
    """
    可视化函数框架（需根据实际数据结构实现）
    参数：
        inputs: 输入序列 [T, V, C]
        preds: 预测序列 [T, V, C]
        targets: 真实序列 [T, V, C]
        save_path: 图片保存路径
    """
    # 示例：绘制第一个关节的X坐标变化
    plt.figure(figsize=(12, 6))

    # 输入序列（通常比预测长）
    input_len = inputs.shape[0]
    plt.plot(range(input_len), inputs[:, 0, 0], label='Input', color='blue')

    # 预测序列
    pred_len = preds.shape[0]
    plt.plot(range(input_len, input_len + pred_len), preds[:, 0, 0], label='Predicted', color='red')

    # 真实序列
    plt.plot(range(input_len, input_len + pred_len), targets[:, 0, 0], label='Ground Truth', color='green')

    plt.title('Motion Prediction Visualization')
    plt.xlabel('Frame')
    plt.ylabel('Joint Position')
    plt.legend()
    plt.savefig(save_path)
    plt.close()


class Evaluator:
    def __init__(self, args):
        if args.work_dir:
            args.work_dir=args.work_dir+datetime.now().strftime('%Y%m%d_%H%M%S')
        self.args = args
        self.device = torch.device(args.device)


        if not os.path.exists(self.args.work_dir):
            os.makedirs(self.args.work_dir)

        # 加载模型
        self.model = self.load_model()
        # 加载数据
        self.test_loader = self.load_data()
        # 损失函数
        self.criterion = torch.nn.MSELoss()
        self.init_test_recorder()

        self.actions=['posing','greeting','sitting','walking','smoking','walkingtogether','phoning','walkingdog','waiting','eating','discussion','purchases','sittingdown','directions','takingphoto',]


    def load_model(self):
        model = MotionGPT(
            d_model=64,
            n_heads=4,
            num_layers=10,
            max_seq_len=50,
            input_dim=6,
            output_dim=6
        ).to(self.device)

        # 加载训练好的权重
        model.load_state_dict(torch.load(self.args.model_path, map_location=self.device))
        model.eval()
        return model

    def load_data(self):
        test_set = Human36mDataset(
            data_dir=self.args.data_dir,
            split='train',
            input_length=50,
            predicted_length=25
        )
        return DataLoader(
            test_set,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=4
        )

    def evaluate(self):
        total_loss = 0
        os.makedirs(self.args.save_dir, exist_ok=True)

        with torch.no_grad():
            for batch_idx, (inputs, targets, label,target_xyz) in enumerate(tqdm(self.test_loader, desc='Evaluating')):
                inputs = inputs.float().to(self.device)
                targets = targets.float().to(self.device)

                # 前向传播
                outputs = self.model(inputs)
                # print(outputs.shape)
                # 计算损失
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                sixd_diff = outputs - targets  # (B, T, V*6)
                # 对最后一维 (V*6) 做 L2 norm => (B, T)
                # 注意：torch.norm(..., dim=-1) 是 sqrt(∑x^2)，如果想要 mean square 可以改成 squared, etc.
                sixd_l2 =torch.norm(sixd_diff, dim=-1).mean(dim=-1) # (B, T)
                # 6D数据转换为3D坐标数据
                predict_xyz=sixd_to_xyz_torch(outputs.cpu())

                diff = predict_xyz - target_xyz  # (B, T, V, 3)
                dist = torch.norm(diff, dim=-1)  # (B, T, V)
                mpjpe_b_t = dist.mean(dim=-1)  # (B, T)


                for i in range(inputs.shape[0]):
                    a_label = int(label[i].item())
                    # 取出该样本的 (T,) mpjpe
                    sample_error = mpjpe_b_t[i]  # shape: (T,)
                    # 转到 CPU，避免后续堆叠时 GPU/CPU 混合
                    sample_error = sample_error.detach().cpu().numpy()
                    self.test_mpjpe[a_label].append(sample_error)

                    # 6D error
                    sample_error_6d = sixd_l2[i].detach().cpu().numpy()  # shape = (T,)
                    self.test_6derr[a_label].append(sample_error_6d)




        self.summarize_results()



        return 0

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

    def visualize_batch(self, inputs, outputs, targets):
        """可视化批次中的第一个样本"""
        # 转换为CPU 数组
        inputs = inputs[0].cpu().numpy()
        outputs = outputs[0].cpu().numpy()
        targets = targets[0].cpu().numpy()




        # 6D数据转换为3D数据


        # 生成保存路径
        save_path = os.path.join(self.args.save_dir, f'prediction_sample.png')
        visualize_predictions(
            inputs=inputs,
            preds=outputs,
            targets=targets,
            save_path=save_path
        )

    def record_mpjpe(self, predict_seq, target_seq, batch_labels):
        if not isinstance(batch_labels, np.ndarray):
            batch_labels = batch_labels.cpu().numpy().reshape(-1).tolist()

        # 获取输入形状 (B, T, V, C)
        B, T, V, C = predict_seq.shape

        # 调整维度顺序为 (B, C, T, V) 以便后续处理
        predict_seq = predict_seq.permute(0, 3, 1, 2)  # [B, C, T, V]
        target_seq = target_seq.permute(0, 3, 1, 2)  # [B, C, T, V]

        # 创建带额外关节的容器
        zero_predict = torch.zeros((B, C, T, V ), device=predict_seq.device)
        zero_target = torch.zeros((B, C, T, V ), device=target_seq.device)

        # 填充数据到前 V 个关节
        zero_predict[:, :, :, :V] = predict_seq
        zero_target[:, :, :, :V] = target_seq

        # 计算 MPJPE (在坐标维度 C 上计算 L2 范数)
        batch_errors = torch.norm(zero_predict - zero_target, p=2, dim=1)  # [B, T, V+1]
        batch_errors = batch_errors.mean(dim=-1).cpu().numpy()  # [B, T]

        # 存储结果
        for idx, action in enumerate(batch_labels):
            self.test_mpjpe[action].extend(batch_errors[idx] )

    def init_test_recorder(self):

        self.test_mpjpe={i:[] for i in range(self.args.num_class)}
        self.test_6derr = {i: [] for i in range(self.args.num_class)}



    def print_log(self, s, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            s = f'[ {localtime} ] {s}'
        print(s)
        if self.args.print_log:
            with open(os.path.join(self.args.work_dir, 'log.txt'), 'a') as f:
                print(s, file=f)

def main():
    parser = get_parser()
    args = parser.parse_args()

    if args.config:
        with open(args.config, 'r') as f:
            default_args = yaml.safe_load(f)
        parser.set_defaults(**default_args)
    args = parser.parse_args()

    evaluator = Evaluator(args)
    evaluator.evaluate()


if __name__ == "__main__":
    main()