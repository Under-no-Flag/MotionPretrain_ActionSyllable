# train_clf.py
# ------------------------------------------------------------
# 训练 H36M 动作分类器（MotionVQVAE ➜ indices ➜ MotionClassifier）
# ------------------------------------------------------------
import os, json, torch, torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.h36m import H36MClfDataset,create_h36m_clf_dataloader          # ← 你把上面那段代码单独保存成 dataset/h36m_clf.py
from model.motion_vqvae   import MotionVQVAE
from model.downstream_models   import MotionClassifier
import utils.utils_model as utm                       # 同 train_mp.py 的日志工具

# ------------------ A. 参数 ------------------
class Args:
    # 数据与文件
    data_root    = '../data/h3.6m'
    split_train  = 'train'
    split_val    = 'val'
    out_dir      = '../checkpoints'
    exp_name     = 'clf_h36m'
    vq_path = '../checkpoints/vqvae/vq_vae_h36m.pth'

    # 训练
    batch_size   = 8
    epochs       = 80
    lr           = 1e-3
    weight_decay = 1e-4
    seed         = 42

    # 模型超参
    # 这些数字要和预训练 VQVAE 对应：
    codebook_size = 256      # vqvae.quantizer.n_e
    n_q           = 1        # 如果你用的是 RVQ>1 就改
    emb_dim       = 128      # vqvae.e_dim
    hidden_dim    = 32
    num_classes   = 15       # Human3.6M 动作类别数量
    dropout       = 0.2

args = Args(); torch.manual_seed(args.seed)
save_dir = os.path.join(args.out_dir, args.exp_name); os.makedirs(save_dir, exist_ok=True)
logger  = utm.get_logger(save_dir);   writer = SummaryWriter(save_dir)
logger.info(json.dumps(vars(args), indent=2))

# ------------------ B. 数据 ------------------
train_set = H36MClfDataset(args.data_root, args.split_train)
val_set   = H36MClfDataset(args.data_root, args.split_val)
train_loader = create_h36m_clf_dataloader(train_set, batch_size=args.batch_size,
                          shuffle=True,pin_memory=True)
val_loader   = create_h36m_clf_dataloader(val_set,   batch_size=args.batch_size,
                           pin_memory=True,shuffle=False)

# ------------------ C. 模型 ------------------
# 1) 预训练 MotionVQVAE
vqvae = MotionVQVAE(
    n_heads=4, num_joints=32, in_dim=6, n_codebook=32,
    balance=0, n_e=args.codebook_size, e_dim=args.emb_dim,
    hid_dim=args.emb_dim, beta=0.25, quant_min_prop=1.0,
    n_layers=[0, 6], seq_len=64
).cuda()

state   = torch.load(args.vq_path, map_location='cpu')
vqvae.load_state_dict(state['net']); vqvae.eval()
# 2) MotionClassifier（仅分类器可训练）
clf = MotionClassifier(vqvae,
                       codebook_size=args.codebook_size,
                       n_q=args.n_q,
                       emb_dim=args.emb_dim,
                       num_classes=args.num_classes,
                       hidden_dim=args.hidden_dim,
                       dropout=args.dropout).cuda()

for p in clf.vqvae.parameters():
    p.requires_grad_(False)

opt   = optim.AdamW(filter(lambda p: p.requires_grad, clf.parameters()),
                    lr=args.lr, weight_decay=args.weight_decay)
sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max',
                                                   factor=0.5, patience=5,
                                                   verbose=True)
criterion = nn.CrossEntropyLoss()

# ------------------ D. 训练 ------------------
best_acc = 0.0
for epoch in range(args.epochs):
    # ---- 1. Train ----
    clf.train(); running_loss = 0; correct = 0; total = 0
    pbar = tqdm(train_loader, desc=f'E{epoch}')
    for x, l,y in pbar:                     # x:(B,T,V,6)  y:(B,)
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        logits = clf(x)                   # (B, num_classes)
        loss   = criterion(logits, y)

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += loss.item() * x.size(0)
        pred  = logits.argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)
        pbar.set_postfix(loss=f'{loss.item():.4f}',
                         acc =f'{correct/total:.3f}')

    train_loss = running_loss / len(train_set)
    train_acc  = correct / total

    # ---- 2. Validate ----
    clf.eval(); val_loss = 0; val_correct = 0; val_total = 0
    with torch.no_grad():
        for x,l, y in val_loader:
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            logits = clf(x)
            val_loss += criterion(logits, y).item() * x.size(0)
            val_correct += (logits.argmax(1) == y).sum().item()
            val_total   += y.size(0)

    val_loss /= len(val_set)
    val_acc  = val_correct / val_total

    # ---- 3. 日志 & 调度 ----
    sched.step(val_acc)
    writer.add_scalar('loss/train', train_loss, epoch)
    writer.add_scalar('loss/val',   val_loss,  epoch)
    writer.add_scalar('acc/train',  train_acc, epoch)
    writer.add_scalar('acc/val',    val_acc,   epoch)
    logger.info(f'Epoch {epoch:03d}: '
                f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
                f'train_acc={train_acc:.3f}  val_acc={val_acc:.3f}')

    # ---- 4. Save best ----
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save({'clf':   clf.state_dict(),
                    'epoch': epoch,
                    'val_acc': val_acc},
                   os.path.join(save_dir, 'best_clf.pth'))
        logger.info(f'New best acc {best_acc:.3f} @ epoch {epoch}')

logger.info('Training finished ')
