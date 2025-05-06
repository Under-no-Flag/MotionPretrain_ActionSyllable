import torch, numpy as np
from tqdm import tqdm
from collections import defaultdict
from dataset.h36m import H36MSeq2Seq
from model.motion_vqvae   import MotionVQVAE
from model.downstream_models import MotionGPT
from utils.forward_kinematics import sixd_to_xyz_torch   # 已有
# ------------------------------------------------------------
actions = [
    'posing','greeting','sitting','walking','smoking','walkingtogether',
    'phoning','walkingdog','waiting','eating','discussion',
    'purchases','sittingdown','directions','takingphoto'
]
num_actions = len(actions)
test_mpjpe = {i: [] for i in range(num_actions)}         # per-action缓存

# =============== A. 运行参数 ===============================
class Args:
    data_root   = './data/h3.6m'
    split_test  = 'test'
    hist_len    = 50
    pred_len    = 25
    batch_size  = 4
    seed        = 42
    vq_vae_path = './checkpoints/vqvae/vq_vae_h36m.pth'
    mpgt_path   = './checkpoints/gpt_seq2seq/best_seq2seq_large.pth'
args = Args(); torch.manual_seed(args.seed)

# =============== B. 构建模型并载权重 =======================
vqvae = MotionVQVAE(
    n_heads=4, num_joints=32, in_dim=6, n_codebook=32,
    balance=0, n_e=256, e_dim=128, hid_dim=128, beta=0.25,
    quant_min_prop=1.0, n_layers=[0, 6], seq_len=64
)
gpt = MotionGPT(vqvae, n_head=4, embed_dim=128, n_layers=6).cuda()

vqvae.load_state_dict(torch.load(args.vq_vae_path)['net'])
gpt.load_state_dict(torch.load(args.mpgt_path)['gpt'])
vqvae.eval().cuda(); gpt.eval()

# =============== C. 数据集 ================================
test_set = H36MSeq2Seq(args.data_root, args.split_test,
                       args.hist_len + args.pred_len, 1.0)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False
)

pred_len = args.pred_len
xyz_scale = 1000.0   # → 毫米
# ================= D. 推理 & 误差 =========================
with torch.no_grad():
    pbar = tqdm(test_loader, desc='MotionGPT MPJPE')
    for x_hist, x_fut, xyz_fut, labels in pbar:           # (B,50,V,6)/(B,25,V,6)/(B,25,V,3)
        x_hist, x_fut, xyz_fut = x_hist.cuda(), x_fut.cuda(), xyz_fut.cuda()
        # ---------- 1. 量化历史帧 indices ------------------
        _, _, idx = gpt.vqvae(x=x_hist, y=0,
                              valid=torch.ones(x_hist.size(0), x_hist.size(1),
                                               device=x_hist.device))
        if idx.dim() == 3: idx = idx.unsqueeze(-1)
        idx_hist = idx                                    # [B,50,V,1]

        # ---------- 2. 自回归预测 25 帧 --------------------
        pred_steps = []
        for _ in range(pred_len):
            mask = gpt.generate_causal_mask(idx_hist.size(1),
                                             gpt.num_joints, idx_hist.device)
            h = gpt.transformer(idx_hist, mask)[:, -1]    # (B,V,D)
            logits0 = torch.stack(
                [gpt.joint_heads[str(j)](h[:, j]) for j in range(gpt.num_joints)], 1
            )                                             # (B,V,K)
            next_idx0 = logits0.argmax(-1)
            pad = idx_hist[:, -1, :, 1:]                  # 其他级别复用
            new_step = torch.cat([next_idx0.unsqueeze(-1), pad], -1)
            idx_hist = torch.cat([idx_hist, new_step.unsqueeze(1)], 1)
            pred_steps.append(new_step)

        pred_idx_future = torch.stack(pred_steps, 1)      # (B,25,V,n_q)

        # ---------- 3. rotmat → 6D → xyz ------------------


        rot_pred = gpt.vqvae.forward_from_indices(pred_idx_future)  # (B,25,V,3,3)
        sixd_pred = rot_pred
        xyz_pred  = sixd_to_xyz_torch(sixd_pred)                     # (B,25,V,3)

        # ---------- 4. MPJPE & 记录 -----------------------
        diff  = (xyz_pred - xyz_fut)             # 到毫米
        mpjpe = torch.norm(diff, dim=-1).mean(-1).cpu().numpy()   # (B,25)

        for i in range(labels.size(0)):
            act_idx = int(labels[i])
            test_mpjpe[act_idx].append(mpjpe[i])

# ================= E. 统计打印 =============================
print('\n{:<16}|'.format('milliseconds'),
      *[' {:5d} |'.format((i+1)*40) for i in range(pred_len)])

avg_error_ms = np.zeros((num_actions, pred_len))
for act_idx, act_name in enumerate(actions):
    if len(test_mpjpe[act_idx]) == 0:
        continue
    per_act = np.stack(test_mpjpe[act_idx]).mean(0)
    avg_error_ms[act_idx] = per_act
    print('{:<16}|'.format(act_name),
          *[' {:5.1f} |'.format(e) for e in per_act])

avg1 = avg_error_ms.mean(0)                     # 先 per-action 再均值
print('{:<16}|'.format('Avg(all actions)'),
      *[' {:5.1f} |'.format(e) for e in avg1])

all_mpjpe = np.concatenate([np.stack(v) for v in test_mpjpe.values()], 0)
avg2 = all_mpjpe.mean(0)                        # 直接全样本均值
print('{:<16}|'.format('Avg(all samples)'),
      *[' {:5.1f} |'.format(e) for e in avg2])
