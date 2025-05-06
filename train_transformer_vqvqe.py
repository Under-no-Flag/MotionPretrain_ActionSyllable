import os
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

import model.vq_vae as vqvae
import utils.losses as losses
import utils.utils_model as utils_model
from dataset.h36m import HumanVQVAESixDDataSet  # 导入修改后的数据集类
from model.transformer_vqvae import TransformerVQVAE  # 导入TransformerVQVAE模型
from options import option_transformer_vqvae  # 导入TransformerVQVAE模型参数配置
from model.motion_vqvae import MotionVQVAE  # 导入MotionVQVAE模型
# 禁用警告
import warnings

warnings.filterwarnings('ignore')


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    """学习率预热函数"""
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


class TrainingMonitor:
    """训练损失跟踪器"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.recon_loss = 0.0
        self.commit_loss = 0.0
        self.geo_loss = 0.0
        self.total_samples = 0

    def update(self, recon, commit, geo, batch_size):
        self.recon_loss += recon * batch_size
        self.commit_loss += commit * batch_size
        self.geo_loss += geo * batch_size
        self.total_samples += batch_size

    def get_avg_losses(self):
        avg_recon = self.recon_loss / self.total_samples
        avg_commit = self.commit_loss / self.total_samples
        avg_geo = self.geo_loss / self.total_samples
        return avg_recon, avg_commit, avg_geo

if __name__ == "__main__":
    ##### ---- 参数配置 ---- #####
    args = option_transformer_vqvae.get_args_parser()
    torch.manual_seed(args.seed)

    # 数据集特定参数
    args.dataname = 'h36m'  # 指定数据集名称
    args.nb_joints = 32  # Human3.6m关节数
    args.out_dir = os.path.join(args.out_dir, f'h36m_{args.exp_name}')
    os.makedirs(args.out_dir, exist_ok=True)

    ##### ---- 日志记录 ---- #####
    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

    ##### ---- 数据加载 ---- #####
    # 训练集
    train_dataset = HumanVQVAESixDDataSet(
        data_dir='./data/h3.6m',
        split='train',
        max_seq_len=args.window_size,
        sixd_scale=1.0
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True
    )

    # 验证集
    val_dataset = HumanVQVAESixDDataSet(
        data_dir='./data/h3.6m',
        split='val',
        max_seq_len=args.window_size,
        sixd_scale=1.0
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    ##### ---- 模型初始化 ---- #####
    # net = TransformerVQVAE(
    #     n_heads=4,
    #     in_dim=6,
    #     num_joints=32,
    #     n_codebook=32,
    #     balance=0,
    #     n_e=512,
    #     e_dim=512,
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
        n_codebook=32,
        balance=0,
        n_e=256,
        e_dim=64,
        hid_dim=64,
        beta=0.25,
        quant_min_prop=1.0,
        n_layers=[0, 10],
        seq_len=64,
    )
    if torch.cuda.is_available():
        net = net.cuda()


    optimizer = optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay
    )
    # 替换为 ReduceLROnPlateau 调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',       # 监控验证损失的最小化
        factor=0.95,       # 学习率衰减比例为0.5
        patience=8,       # 3个epoch无改善后调整学习率
        verbose=True      # 打印学习率调整日志
    )

    ##### ---- 损失函数 ---- #####
    loss_func = losses.ReConsLoss(
        recons_loss=args.recons_loss,
    )
    geo_loss_func=losses.GeometricConstraintLoss()

    ##### ---- 训练循环 ---- #####
    best_fid = 1000
    for epoch in range(args.total_epoch):
        monitor = TrainingMonitor()
        net.train()
        avg_recons, avg_commit = 0.0, 0.0

        progress_bar = tqdm(enumerate(train_loader), dynamic_ncols=True, desc=f"Training Epoch {epoch}",
                            total=len(train_loader))

        for batch_idx, (data, _, _) in progress_bar:
            if torch.cuda.is_available():
                data = data.cuda().float()

            # 前向传播
            (pred_motion,), quant_item, _ = net(x=data, y=0,
                                                valid=torch.ones(data.shape[0], data.shape[1]).to(data.device))

            # 计算损失
            loss_commit = quant_item['quant_loss']
            loss_recons = loss_func(pred_motion, data)
            loss = loss_recons + args.commit * loss_commit

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新进度条的损失信息（保留4位小数）
            progress_bar.set_postfix(
                recons_loss=f"{loss_recons.item():.4f}",
                commit_loss=f"{loss_commit.item():.4f}",
                total_loss=f"{loss.item():.4f}"
            )

            # 记录到监控器（假设monitor处理累积或日志）
            batch_size = data.size(0)
            monitor.update(
                loss_recons.item(),
                loss_commit.item(),
                0,
                batch_size
            )



        # 验证阶段
        net.eval()
        with torch.no_grad():
            val_recons = 0.0
            for val_data,_,_ in val_loader:
                if torch.cuda.is_available():
                    val_data = val_data.cuda().float()

                (recon_val,), quant_item, _ = net(x=val_data, y=0,
                                                    valid=torch.ones(val_data.shape[0], val_data.shape[1]).to(val_data.device))
                val_recons += loss_func(recon_val, val_data).item()

            val_recons /= len(val_loader)
            logger.info(f"Validation Loss: {val_recons:.4f}")
            writer.add_scalar('Val/Recons', val_recons, epoch)

            scheduler.step(val_recons)
            #print learning rate
            for param_group in optimizer.param_groups:
                logger.info(f"Learning Rate: {param_group['lr']:.6f}")

            # 保存最佳模型
            if val_recons < best_fid:
                best_fid = val_recons
                torch.save({
                    'net': net.state_dict(),
                    'epoch': epoch,
                    'loss': val_recons
                }, os.path.join(args.out_dir, 'best_model.pth'))

        avg_recon, avg_commit, avg_geo = monitor.get_avg_losses()
        logger.info(f"Epoch {epoch} Average: "
              f"Recon: {avg_recon:.4f} "
              f"Commit: {avg_commit:.4f} "
              f"Geo: {avg_geo:.4f} \n")


    logger.info("Training Completed!")