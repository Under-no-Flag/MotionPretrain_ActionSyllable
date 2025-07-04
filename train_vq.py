import os
import json
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import model.vq_vae as vqvae
import utils.losses as losses
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset.h36m import HumanVQVAESixDDataSet  # 导入修改后的数据集类


# 禁用警告
import warnings

warnings.filterwarnings('ignore')


def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):
    """学习率预热函数"""
    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return optimizer, current_lr


if __name__ == "__main__":
    ##### ---- 参数配置 ---- #####
    args = option_vq.get_args_parser()
    torch.manual_seed(args.seed)

    # 数据集特定参数
    args.dataname = 'h36m'  # 指定数据集名称
    args.nb_joints = 32  # Human3.6m关节数
    args.window_size = 64  # 输入序列长度
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
    net = vqvae.HumanVQVAE(
        args,
        nb_code=args.nb_code,
        code_dim=args.code_dim,
        output_emb_width=args.output_emb_width,
        down_t=args.down_t,
        stride_t=args.stride_t,
        width=args.width,
        depth=args.depth,
        dilation_growth_rate=args.dilation_growth_rate,
        activation=args.vq_act,
        norm=args.vq_norm,
    )

    if torch.cuda.is_available():
        net = net.cuda()

    ##### ---- 优化器配置 ---- #####
    optimizer = optim.AdamW(
        net.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=args.lr_scheduler,
        gamma=args.gamma
    )

    ##### ---- 损失函数 ---- #####
    loss_func = losses.ReConsLoss(
        recons_loss=args.recons_loss,
        nb_joints=args.nb_joints
    )

    ##### ---- 训练循环 ---- #####
    best_fid = 1000
    for epoch in range(args.total_epoch):
        net.train()
        avg_recons, avg_commit = 0.0, 0.0

        # 训练阶段
        for batch_idx, (data,_,_) in enumerate(train_loader):
            if torch.cuda.is_available():
                data = data.cuda().float()  # (B, V*C, T)

            # 前向传播
            pred_motion, loss_commit, _ = net(data)

            # 计算损失
            loss_recons = loss_func(pred_motion, data)
            loss = loss_recons + args.commit * loss_commit

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 记录损失
            avg_recons += loss_recons.item()
            avg_commit += loss_commit.item()

            # 日志记录
        if epoch % args.print_epoch == 0:
            avg_recons /= args.print_epoch
            avg_commit /= args.print_epoch

            logger.info(
                f"Epoch:{epoch}  "
                f"Recons:{avg_recons:.4f} Commit:{avg_commit:.4f}"
            )
            writer.add_scalar('Train/Recons', avg_recons, epoch)
            writer.add_scalar('Train/Commit', avg_commit, epoch)

            avg_recons, avg_commit = 0.0, 0.0

        # 验证阶段
            net.eval()
            with torch.no_grad():
                val_recons = 0.0
                for val_data,_,_ in val_loader:
                    if torch.cuda.is_available():
                        val_data = val_data.cuda().float()

                    recon_val, _, _ = net(val_data)
                    val_recons += loss_func(recon_val, val_data).item()

                val_recons /= len(val_loader)
                logger.info(f"Validation Loss: {val_recons:.4f}")
                writer.add_scalar('Val/Recons', val_recons, epoch)

                # 保存最佳模型
                if val_recons < best_fid:
                    best_fid = val_recons
                    torch.save({
                        'net': net.state_dict(),
                        'epoch': epoch,
                        'loss': val_recons
                    }, os.path.join(args.out_dir, 'best_model.pth'))

        scheduler.step()

    logger.info("Training Completed!")