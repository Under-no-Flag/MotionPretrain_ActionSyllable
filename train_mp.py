# ===========================  train_mp_sw.py  ===========================
import argparse, os, json, yaml, atexit, signal, sys, torch, torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import swanlab                                                   # pip install swanlab

from dataset.h36m import H36MSeq2Seq
from model.motion_vqvae import MotionVQVAE
from model.downstream_models import MotionGPT
import utils.utils_model as utm


# --------------------------- 辅助函数 ----------------------------------
def _dotdict(d: dict):
    return SimpleNamespace(**d)


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return _dotdict(yaml.safe_load(f))


def _register_swanlab_exit():
    """确保 Ctrl-C / kill 时能正常 swanlab.finish()"""
    def _graceful(_sig=None, _frm=None):
        swanlab.finish(); sys.exit(0)

    atexit.register(swanlab.finish)
    signal.signal(signal.SIGINT,  _graceful)
    signal.signal(signal.SIGTERM, _graceful)


# ======================================================================
def main():
    # ------------------------- CLI / YAML -----------------------------
    parser = argparse.ArgumentParser(
        description="MotionGPT seq-to-seq training with SwanLab tracking")
    parser.add_argument("--config", type=str, required=True,
                        help="path to yaml config")
    cli = parser.parse_args()
    cfg = load_config(cli.config)          # -> cfg.xxx

    # ------------------ reproducibility -------------------------------
    torch.manual_seed(cfg.seed)

    # ------------------ logging dirs ----------------------------------
    out_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(out_dir, exist_ok=True)
    writer  = SummaryWriter(out_dir)
    logger  = utm.get_logger(out_dir)
    logger.info("Loaded cfg:\n" + json.dumps(vars(cfg), indent=2))

    # ------------------ SwanLab ---------------------------------------
    swanlab.login(api_key=cfg.api_key, save=True)
    sw_run = swanlab.init(project=cfg.project_name,
                          name=cfg.exp_name,
                          config=vars(cfg))
    # _register_swanlab_exit()      # 可按需开启

    # ------------------ dataset ---------------------------------------
    train_set = H36MSeq2Seq(cfg.data_root, cfg.split_train,
                            cfg.hist_len + cfg.pred_len, 1.0)
    val_set   = H36MSeq2Seq(cfg.data_root, cfg.split_val,
                            cfg.hist_len + cfg.pred_len, 1.0)

    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set,   batch_size=cfg.batch_size, shuffle=False)

    # ------------------ model -----------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ① 冻结好的 VQVAE
    vqvae = MotionVQVAE(**cfg.vqvae).to(device)
    vq_ckpt = torch.load(cfg.vqvae_ckpt, map_location=device)
    vqvae.load_state_dict(vq_ckpt["net"]); vqvae.eval()
    for p in vqvae.parameters(): p.requires_grad_(False)

    # ② MotionGPT
    gpt = MotionGPT(vqvae,
                    n_ctx=cfg.seq_len,
                    n_head=cfg.n_head,
                    embed_dim=cfg.embed_dim,
                    n_layers=cfg.n_layers).to(device)

    opt   = optim.AdamW(gpt.parameters(), lr=cfg.lr,
                        weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=cfg.lr_patience, verbose=True)
    ce_fn = nn.CrossEntropyLoss(label_smoothing=cfg.label_smooth)
    K     = gpt.codebook_sz

    # ------------------ training loop ---------------------------------
    best_val = 1e9
    global_step = 0

    for epoch in range(cfg.epochs):
        # ===== Train ==================================================
        gpt.train(); train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for x_hist, x_fut, _ in pbar:
            x_hist, x_fut = x_hist.to(device), x_fut.to(device)

            logits, tgt = gpt(x_hist,
                              pred_len=cfg.pred_len,
                              teacher_force=True,
                              future_seq=x_fut)          # [B,Tp,V,n_q,K]

            loss = ce_fn(logits.reshape(-1, K), tgt.reshape(-1))

            opt.zero_grad(); loss.backward(); opt.step()
            train_loss += loss.item() * x_hist.size(0)

            if global_step % cfg.log_interval == 0:
                swanlab.log({"train/ce": loss.item(),
                             "lr": opt.param_groups[0]["lr"]},
                            step=global_step)
            global_step += 1
            pbar.set_postfix(ce=f"{loss.item():.4f}")

        train_ce = train_loss / len(train_set)

        # ===== Validation ============================================
        gpt.eval(); val_loss = 0.0
        with torch.no_grad():
            for x_hist, x_fut, _ in tqdm(val_loader):
                x_hist, x_fut = x_hist.to(device), x_fut.to(device)
                logits, tgt = gpt(x_hist, cfg.pred_len,
                                  teacher_force=True,
                                  future_seq=x_fut)
                val_loss += ce_fn(logits.reshape(-1, K),
                                  tgt.reshape(-1)).item() * x_hist.size(0)
        val_ce = val_loss / len(val_set)

        # ---- Scheduler & Log ---------------------------------------
        sched.step(val_ce)
        writer.add_scalar("train/ce", train_ce, epoch)
        writer.add_scalar("val/ce",   val_ce,   epoch)
        swanlab.log({"val/ce": val_ce}, step=global_step)
        logger.info(f"Epoch {epoch}: train_ce={train_ce:.4f}  val_ce={val_ce:.4f}")

        # ---- Save best --------------------------------------------
        if val_ce < best_val:
            best_val = val_ce
            ckpt_path = os.path.join(out_dir, "best_seq2seq.pth")
            torch.save({"gpt": gpt.state_dict(),
                        "epoch": epoch,
                        "val_ce": val_ce},
                       ckpt_path)
            swanlab.log({"best/val_ce": best_val}, step=global_step)
            logger.info(f"New best saved to {ckpt_path}")

    # ------------------ finish ----------------------------------------
    swanlab.finish()
    logger.info(f"Finished. Best val CE = {best_val:.4f}")


if __name__ == "__main__":
    main()
