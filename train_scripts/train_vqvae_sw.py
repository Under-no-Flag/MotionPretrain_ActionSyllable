import argparse
import json
import os
import warnings
from types import SimpleNamespace
import signal
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import yaml
import swanlab  # pip install swanlab
import atexit
import importlib
print(os.getcwd())
import utils.losses as losses
import utils.utils_model as utils_model
from model.motion_vqvae import MotionVQVAE
from options.option_vq import get_args_parser
from utils.func import import_class
from torch import nn
# --------------------------------------------------------  Codebook monitor
def log_codebook_usage(model: nn.Module, writer, step: int, to_swanlab=True):
    """
    遍历模型里所有 ResidualVectorQuantizer，记录每级 codebook 的
    ① 激活比例 active_ratio (= 被选中次数占比 >1e-3 的 code 数)
    ② 困惑度 perplexity (= exp(熵))
    ③ 直方图 hist
    """
    for k, tracker in model.quantizer.rvq.trackers.items():
        hist = tracker.get_hist().detach().cpu()          # (n_e,)
        active_ratio = (hist > 1e-3).float().mean().item()
        perplexity = torch.exp(
            -(hist * (hist + 1e-8).log()).sum()
        ).item()

        tag = f"stage{k}"
        # ---- TensorBoard
        writer.add_scalar(f"{tag}/active_ratio", active_ratio, step)
        writer.add_scalar(f"{tag}/perplexity",  perplexity,  step)
        writer.add_histogram(f"{tag}/hist", hist,            step)

        # ---- SwanLab
        if to_swanlab:
            swanlab.log(
                {
                    f"train/{tag}_active": active_ratio,
                    f"train/{tag}_perplex": perplexity,
                },
                step=step,
            )


def _register_swanlab_exit():
    """Make sure swanlab.finish() is always executed."""

    def _graceful(_sig=None, _frm=None):
        # flush TensorBoard writers etc. if needed here
        swanlab.finish()
        sys.exit(0)

    # 1) normal program exit
    atexit.register(swanlab.finish)
    # 2) SIGINT ^C and SIGTERM (e.g. kill or job stop)
    signal.signal(signal.SIGINT, _graceful)
    signal.signal(signal.SIGTERM, _graceful)

class TrainingMonitor:
    """Accumulates epoch‑level averages for losses."""

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
        if self.total_samples == 0:
            return 0.0, 0.0, 0.0
        return (
            self.recon_loss / self.total_samples,
            self.commit_loss / self.total_samples,
            self.geo_loss / self.total_samples,
        )


def _dotdict(d: dict):
    """dict -> argparse.Namespace"""
    return SimpleNamespace(**d)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)
    return _dotdict(cfg_dict)




def main():
    parser = argparse.ArgumentParser(description="Train Motion-VQVAE with SwanLab tracking")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    cli_args = parser.parse_args()
    args = load_config(cli_args.config)

    # ------------------------------------------------------------------ setup
    torch.manual_seed(args.seed)
    args.out_dir = os.path.join(args.out_dir, f"{args.dataset}_{args.exp_name}")
    os.makedirs(args.out_dir, exist_ok=True)

    logger = utils_model.get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    logger.info("Loaded configuration:\n" + json.dumps(vars(args), indent=4))
    swanlab.login(api_key=args.api_key, save=True)
    sl_run = swanlab.init(project=args.project_name, name=args.exp_name, config=vars(args))
    # _register_swanlab_exit()
    # ------------------------------------------------------------ data loaders
    # 动态导入数据集类
    DatasetClass = import_class(args.dataset_class)
    
    train_dataset = DatasetClass(
        data_dir=args.data_root,
        split="train",
        max_seq_len=args.window_size,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    val_dataset = DatasetClass(
        data_dir=args.data_root,
        split="val",
        max_seq_len=args.window_size,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # ---------------------------------------------------------------- model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = MotionVQVAE(
        dataset_name= args.dataset,
        n_heads=args.n_heads,
        num_joints= args.nb_joints,
        in_dim= args.in_dim,
        n_codebook=args.n_codebook,
        balance=args.balance,
        n_e=args.n_e,
        e_dim=args.e_dim,
        hid_dim=args.hid_dim,
        beta= args.beta,
        quan_min_pro= args.quant_min_prop,
        n_layers=args.n_layers,
        seq_len=args.seq_len,
    ).to(device)

    if args.model_path:
        ckpt = torch.load(args.model_path, map_location=device)
        net.load_state_dict(ckpt["net"])
        logger.info(f"Loaded checkpoint from {args.model_path}")

    optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.95, patience=args.lr_patience, verbose=True
    )

    loss_func = losses.ReConsLoss(recons_loss=args.recons_loss)
    monitor = TrainingMonitor()

    global_step = 0
    best_val = args.best_init

    # ---------------------------------------------------------------- training
    for epoch in range(args.total_epoch):
        net.train()
        monitor.reset()

        pbar = tqdm(train_loader, dynamic_ncols=True, desc=f"Epoch {epoch}")
        for batch_idx, (data, _, _) in enumerate(pbar):
            data = data.to(device).float()
            (pred_motion,), quant_item, _ = net(x=data, y=0, valid=torch.ones(data.shape[0], data.shape[1]).to(data.device))

            loss_recons = loss_func(pred_motion, data)
            loss_commit = quant_item["quant_loss"]
            loss = loss_recons + args.commit * loss_commit

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            monitor.update(loss_recons.item(), loss_commit.item(), 0, data.size(0))

            if global_step % args.log_interval == 0:
                swanlab.log(
                    {
                        "train/recon_loss": loss_recons.item(),
                        "train/commit_loss": loss_commit.item(),
                        "train/total_loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )

            global_step += 1
            log_codebook_usage(net, writer, global_step)
            pbar.set_postfix(recon=f"{loss_recons.item():.6f}", commit=f"{loss_commit.item():.6f}")

        # ---------------------------------------------------- validation
        net.eval()
        val_recons = 0.0
        with torch.no_grad():
            for val_data, _, _ in val_loader:
                val_data = val_data.to(device).float()
                (recon_val,), _, _ = net(x=val_data, y=0, valid=torch.ones(val_data.shape[0], val_data.shape[1]).to(val_data.device))
                val_recons += loss_func(recon_val, val_data).item()
        val_recons /= len(val_loader)

        swanlab.log({"val/recon_loss": val_recons}, step=global_step)
        writer.add_scalar("Val/Recons", val_recons, epoch)
        scheduler.step(val_recons)
        avg_recon, avg_commit, _ = monitor.get_avg_losses()
        logger.info(
            f"Epoch {epoch}: train recon {avg_recon:.6f}, train commit {avg_commit:.6f}, val recon {val_recons:.6f}"
        )

        if val_recons < best_val:
            best_val = val_recons
            ckpt_path = os.path.join(args.out_dir, "best_model.pth")
            torch.save({"net": net.state_dict(), "epoch": epoch, "loss": val_recons}, ckpt_path)
            swanlab.log({"best/val_recon_loss": best_val}, step=global_step)
            logger.info(f"New best model saved to {ckpt_path}")

    # ------------------------------------------------------------- finish
    swanlab.finish()
    logger.info("Training completed Best validation recon loss = %.6f", best_val)


if __name__ == "__main__":
    main()