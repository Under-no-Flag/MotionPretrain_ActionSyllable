import os, json, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.h36m import H36MSeq2Seq
from model.motion_vqvae   import MotionVQVAE
from model.downstream_models    import MotionGPT
import utils.utils_model  as utm

# ------------------ A. 参数 ------------------
class Args:
    data_root     = './data/h3.6m'
    split_train   = 'train'
    split_val     = 'val'
    hist_len      = 50
    pred_len      = 25
    batch_size    = 16
    epochs        = 100
    lr            = 2e-3
    weight_decay  = 1e-4
    exp_name      = 'gpt_seq2seq'
    out_dir       = './checkpoints'
    seed          = 42
    label_smooth  = 0.0

args = Args(); torch.manual_seed(args.seed)
save_dir = os.path.join(args.out_dir, args.exp_name); os.makedirs(save_dir, exist_ok=True)
logger = utm.get_logger(save_dir);  writer = SummaryWriter(save_dir)
logger.info(json.dumps(vars(args), indent=2))

# ------------------ B. 数据 ------------------
train_set = H36MSeq2Seq(args.data_root, args.split_train,
                        args.hist_len+args.pred_len, 1.0)
val_set   = H36MSeq2Seq(args.data_root, args.split_val,
                        args.hist_len+args.pred_len, 1.0)
train_loader = DataLoader(train_set, batch_size=args.batch_size,
                          shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=args.batch_size)

# ------------------ C. 模型 ------------------
vqvae = MotionVQVAE(
    n_heads=4, num_joints=32, in_dim=6, n_codebook=32,
    balance=0, n_e=256, e_dim=128, hid_dim=128, beta=0.25,
    quant_min_prop=1.0, n_layers=[0, 6], seq_len=64
)

# 加载vqvae的预训练权重
vqvae.load_state_dict(torch.load('./checkpoints/vqvae/vq_vae_h36m.pth')['net'])
vqvae.eval();
vqvae.cuda()

gpt = MotionGPT(vqvae,
                n_head=4,
                embed_dim=32,
                n_layers=1,
                ).cuda()
for p in gpt.vqvae.parameters(): p.requires_grad_(False)

opt = optim.AdamW(filter(lambda p: p.requires_grad, gpt.parameters()),
                  lr=args.lr, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, 'min', 0.5, 5, verbose=True)
ce = nn.CrossEntropyLoss(label_smoothing=args.label_smooth)

# ------------------ D. 训练 ------------------
best = 1e9
for epoch in range(args.epochs):
    gpt.train(); total_loss = 0
    pbar = tqdm(train_loader, desc=f'E{epoch}')
    for x_hist, x_fut, _ in pbar:
        x_hist, x_fut = x_hist.cuda(), x_fut.cuda()   # (B,Thist,V,6) / (B,Tpred,V,6)
        # 1) 量化历史+未来，拿到 code indices
        with torch.no_grad():
            _,_, idx = gpt.vqvae(
                x=torch.cat([x_hist, x_fut], dim=1),  # (B,Tall,V,6)
                y=0, valid=torch.ones(x_hist.size(0), x_hist.size(1)+x_fut.size(1)).cuda()
            )
        if idx.dim()==3: idx = idx.unsqueeze(-1)      # (B,T,V,1)
        # -------- 2) 构造 LM 输入 / 目标 ----------------------
        inp = idx[:, :-1]  # [B,T-1,V,1]  ⟵ 位置 t=0..T-2
        tgt_lvl0 = idx[:, 1:, :, 0]  # [B,T-1,V]    ⟵ 仅第 0 级
        B, Tm1, V, _ = inp.shape

        # loss 只计算在 “未来帧” (t ≥ hist_len) 位置
        future_mask = torch.zeros_like(tgt_lvl0, dtype=torch.bool)
        future_mask[:, args.hist_len - 1:] = True  # 对应 inp 的索引

        # -------- 3) Transformer 前向一次 -------------------
        attn_mask = gpt.generate_causal_mask(Tm1, V, inp.device)
        h = gpt.transformer(inp, attention_mask=attn_mask)  # [B,T-1,V,D]

        logits_lvl0 = torch.stack(  # [B,T-1,V,K]
            [gpt.joint_heads[str(j)](h[..., j, :]) for j in range(V)], 2
        )

        loss = ce(logits_lvl0[future_mask], tgt_lvl0[future_mask])

        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * x_hist.size(0)
        pbar.set_postfix(loss=f'{loss.item():.4f}')


    # ------------------ E. 验证 ------------------
    gpt.eval(); val_loss = 0
    with torch.no_grad():
        for x_hist, x_fut, _ in val_loader:
            x_all = torch.cat([x_hist, x_fut], 1).cuda()
            _,_, idx = gpt.vqvae(
                x=x_all,  # (B,Tall,V,6)
                y=0, valid=torch.ones(x_hist.size(0), x_hist.size(1)+x_fut.size(1)).cuda()
            )
            if idx.dim()==3: idx = idx.unsqueeze(-1)
            inp      = idx[:, :-1]
            tgt_lvl0 = idx[:, 1:, :, 0]
            future_mask = torch.zeros_like(tgt_lvl0, dtype=torch.bool)
            future_mask[:, args.hist_len-1:] = True

            attn_mask = gpt.generate_causal_mask(inp.size(1), V, inp.device)
            h = gpt.transformer(inp, attention_mask=attn_mask)
            logits_lvl0 = torch.stack(
                [gpt.joint_heads[str(j)](h[..., j, :]) for j in range(V)], 2
            )

            val_loss += ce(logits_lvl0[future_mask],
                           tgt_lvl0[future_mask]).item() * x_all.size(0)

    val_loss /= len(val_set)
    sched.step(val_loss)
    writer.add_scalar('train/ce', total_loss/len(train_set), epoch)
    writer.add_scalar('val/ce',   val_loss,                  epoch)
    logger.info(f'Epoch {epoch}: train={total_loss/len(train_set):.4f} val={val_loss:.4f}')

    if val_loss < best:
        best = val_loss
        torch.save({'gpt': gpt.state_dict(),
                    'epoch': epoch, 'val': val_loss},
                   os.path.join(save_dir, 'best_seq2seq.pth'))
logger.info('Done.')
