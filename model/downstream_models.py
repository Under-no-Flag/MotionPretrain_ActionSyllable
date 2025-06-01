import time

from torch import nn
from model.motion_vqvae import MotionVQVAE
import torch


class MotionGPT(nn.Module):
    def __init__(self, vqvae: "MotionVQVAE",
                 n_ctx: int = 512, n_head: int = 8,
                 embed_dim: int = 64, n_layers: int = 6):
        super().__init__()
        self.vqvae        = vqvae.eval()             # 推理用，默认冻结
        self.codebook_sz  = vqvae.quantizer.n_e
        self.n_q          = getattr(vqvae.quantizer, "n_q", 1)
        self.num_joints   = 32

        self.transformer  = StandardTransformer(
            embed_dim=embed_dim,
            n_heads=n_head,
            n_layers=n_layers,
            codebook_size=self.codebook_sz,
            num_joints=self.num_joints,
            n_q=self.n_q
        )

        # 预测 **第 0 级** codeword 的头
        self.head = nn.Linear(embed_dim, self.codebook_sz * self.n_q, bias=True)



    # ============== 每帧并行算 n_q 级 logits ==============

    def _level_logits(self, h):  # h: [B,V,D]
        logits = self.head(h)  # [B,V,K*n_q]
        return logits.view(*h.shape[:2], self.n_q, self.codebook_sz)
        #                      ↓  ↓              ↑        ↑
        #                  stage  axis-----------╯        ╰-- codeword axis

        # ---------------------------------------------------------
    def forward(
            self,
            x: torch.Tensor,  # 历史序列 [B,T_hist,V,C]
            pred_len: int = 30,  # 需要预测多少帧
            teacher_force: bool = False,  # 训练 = True，推理 = False
            future_seq=None  # 训练时提供 [B,pred_len,V,n_q]
    ):

        """
        x: 输入序列 [B,T0,V,C]  —— 生成长度 max_len
        Return: reconstructed motion [B,max_len,V,C]
        """
        B, T0, V, C= x.size()
        device = x.device

        with torch.no_grad():
            _, _, indices = self.vqvae(x=x, y=0,
                                       valid=torch.ones(x.size(0), x.size(1),
                                                        device=device))

            if teacher_force and future_seq is not None:
                _, _, fut_indices = self.vqvae(x=future_seq, y=0,
                                           valid=torch.ones(future_seq.size(0), future_seq.size(1),
                                                            device=device))

        if indices.dim() == 3:                       # → [B,T,V,1]
            indices = indices.unsqueeze(-1)
            fut_indices = fut_indices.unsqueeze(-1) if teacher_force and future_seq is not None else None

        if teacher_force and fut_indices is not None:
            input_idx = torch.cat([indices, fut_indices[:, :-1]], dim=1)  # hist + (N-1)
            T_full = input_idx.size(1)
            mask = self.generate_causal_mask(T_full, self.num_joints, device)
            h = self.transformer(input_idx, attention_mask=mask)
            h_future = h[:, -pred_len:]
            logits = torch.stack([self._level_logits(h_future[:, t])
                                   for t in range(pred_len)], dim=1)

            return logits,fut_indices

        # ------------------------------------------------------------------ #
        # B. 推理：自回归生成 pred_len 帧
        # ------------------------------------------------------------------ #
        for _ in range(pred_len):
            T_ctx = indices.size(1)
            mask = self.generate_causal_mask(T_ctx, self.num_joints, device)

            h = self.transformer(indices, attention_mask=mask)  # [B,T,V,D]
            last_h = h[:, -1]                                   # [B,V,D]

            logits_all = self._level_logits(last_h)             # [B,V,K,n_q]

            # 采样 / argmax
            new_step = torch.empty(
                indices.size(0), self.num_joints, self.n_q,
                dtype=torch.long, device=device
            )                                                   # [B,V,n_q]
            for q in range(self.n_q):
                new_step[..., q] = torch.distributions.Categorical(
                    logits=logits_all[..., q]).sample()

            indices = torch.cat([indices, new_step.unsqueeze(1)], dim=1)

        # 解码到 6-D rotmat / xyz
        with torch.no_grad():
            motion = self.vqvae.forward_from_indices(indices[:, -pred_len:])
        return motion

    @staticmethod
    def generate_causal_mask(T: int, V: int, device):
        """时空因果掩码 (同帧内全连，跨帧因果)"""
        S = T * V
        mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=device))
        for t in range(T):
            s, e = t * V, (t + 1) * V
            mask[s:e, s:e] = True                    # 同帧允许互看
        return mask


class StandardTransformer(nn.Module):
    def __init__(self, *, embed_dim: int, n_heads: int,
                 n_layers: int, codebook_size: int,
                 num_joints: int, n_q: int):
        super().__init__()
        self.embed_dim   = embed_dim
        self.codebook_sz = codebook_size
        self.num_joints  = num_joints
        self.n_q         = n_q                      # RVQ stages

        # —— 1. 共享 codeword 嵌入（节省参数） ————————————
        self.code_emb = nn.Embedding(codebook_size, embed_dim)
        self.stage_emb = nn.Embedding(self.n_q, embed_dim)

        # —— 2. 时空位置编码 ————————————————————
        self.time_pos  = nn.Embedding(512, embed_dim)        # 支持 ≤512 帧
        self.joint_pos = nn.Embedding(num_joints, embed_dim)

        # —— 3. 编码器 ————————————————————————
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            batch_first=True,
            norm_first=True,
            activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

    # ---------------------------------------------------------
    def forward(self, indices: torch.Tensor, attention_mask: torch.Tensor):
        """
        indices: [B,T,V]  或  [B,T,V,n_q]
        attention_mask: bool, shape [T*V, T*V]  (True=keep, False=mask)
        """
        if indices.dim() == 3:                       # 兼容旧版 VQ
            indices = indices.unsqueeze(-1)          # [B,T,V,1]
        B, T, V, n_q = indices.shape
        device = indices.device

        # —— 1. codeword embedding + mean over n_q ——
        stage_ids = torch.arange(self.n_q, device=device).view(1, 1, 1, self.n_q)
        emb = self.code_emb(indices) + self.stage_emb(stage_ids)  # [B,T,V,n_q,D]
        emb = emb.view(B, T, V * self.n_q, self.embed_dim)  # 或者直接 flatten

        # —— 2. add time / joint positional ——————————
        t_ids = torch.arange(T, device=device).view(1, T, 1)
        v_ids = torch.arange(V, device=device).repeat_interleave(self.n_q)  # [V*n_q]
        v_ids = v_ids.view(1, 1, V * self.n_q)
        emb  = emb + self.time_pos(t_ids) + self.joint_pos(v_ids)  # 广播
        if attention_mask.size(0) == T * V:  # 原 mask
            attention_mask = attention_mask.repeat_interleave(self.n_q, 0)
            attention_mask = attention_mask.repeat_interleave(self.n_q, 1)

        # —— 3. flatten → Transformer ————————————
        x = emb.view(B, T * V * self.n_q, self.embed_dim)        # [B,TV,D]
        x = self.encoder(x, mask=~attention_mask)    # 注意 PyTorch: True=keep

        # —— 4. reshape back ————————————————
        return x.view(B, T, V, self.embed_dim)       # [B,T,V,D]


class MotionClassifier(nn.Module):
    """
    将 MotionVQVAE 的离散索引序列映射为动作类别
    """
    def __init__(
        self,
        vq_vae,
        codebook_size: int,     # 每个 codebook 的大小 n_e
        n_q: int,               # RVQ 级数；单级设为 1
        emb_dim: int,           # 嵌入维度
        num_classes: int,       # 动作类别数
        hidden_dim: int = 128,  # MLP 隐层
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_q = n_q
        self.token_emb = nn.Embedding(codebook_size, emb_dim)

        self.proj = nn.Linear(emb_dim, emb_dim)  # 可选映射

        self.classifier = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.vqvae=vq_vae


    def forward(self, x: torch.Tensor):
        """
        indices:
            [B, T, V]        —— 单级 VQ
            [B, T, V, n_q]   —— 多级 RVQ
        """
        device = x.device
        with torch.no_grad():
            _, _, indices = self.vqvae(x=x, y=0,
                                       valid=torch.ones(x.size(0), x.size(1),
                                                        device=device))
        if indices.dim() == 3:                  # 单级 → 补 n_q 维
            indices = indices.unsqueeze(-1)     # [B,T,V,1]

        # 1) lookup embedding
        emb = self.token_emb(indices)           # [B,T,V,Q,emb_dim]
        # 2) 聚合 RVQ 级数
        emb = emb.mean(-2)                      # [B,T,V,emb_dim]
        # 3) 关节聚合
        emb = emb.mean(-2)                      # [B,T,emb_dim]
        # 4) 时间聚合
        emb = emb.mean(1)                       # [B,emb_dim]
        # 5) 分类
        feat = self.proj(emb)
        logits = self.classifier(feat)
        return logits



if __name__=="__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vqvae = MotionVQVAE(
        n_heads=4,
        num_joints=32,
        in_dim=6,
        n_codebook=8,
        balance=0,
        n_e=256,
        e_dim=128,
        hid_dim=128,
        beta=0.25,
        quant_min_prop=1.0,
        n_layers=[0, 10],
        seq_len=64,
    ).to(device)

    gpt = MotionGPT(vqvae,
                    n_head=4,
                    embed_dim=32,
                    n_layers=3,
                    ).to(device)
    # 计算模型参数量
    num_params = sum(p.numel() for p in gpt.parameters() if p.requires_grad)
    print(f"模型参数量: {num_params / 1e6:.2f}M")
    # 输入初始化序列
    init_motion = torch.randn(16, 50, 32, 6).to(device)  # 10帧初始化


    #计算耗时
    start_time = time.time()
    # 自回归生成未来30帧
    with torch.no_grad():
        generated = gpt(init_motion, pred_len=25)

    print(f"生成运动序列形状: {generated.shape}")  # (30帧, 关节数, 3D坐标)
    # 计算耗时
    end_time = time.time()
    print(f"推理生成耗时: {end_time - start_time:.4f}秒")

    start_time = time.time()
    y=torch.randn(16, 25, 32, 6).to(device)
    gpt.train()
    start_time = time.time()
    generated,_ = gpt(init_motion, pred_len=25,teacher_force=True,future_seq=y)
    end_time = time.time()
    print(f"生成运动序列形状: {generated.shape}")
    print(f"训练生成耗时: {end_time - start_time:.6f}秒")


    # device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    # vqvae = MotionVQVAE(
    #     n_heads=4,
    #     num_joints=32,
    #     in_dim=6,
    #     n_codebook=32,
    #     balance=0,
    #     n_e=64,
    #     e_dim=64,
    #     hid_dim=64,
    #     beta=0.25,
    #     quant_min_prop=1.0,
    #     n_layers=[0, 10],
    #     seq_len=64,
    # ).to(device)
    #
    # clf = MotionClassifier(
    #     vqvae,
    #     codebook_size=128,
    #     n_q=8,
    #     emb_dim=128,
    #     num_classes=10,
    # ).to(device)
    # # 输入初始化序列
    # init_motion = torch.randn(16, 64, 32, 6).to(device)  # 10帧初始化
    #
    #
    # #计算耗时
    # start_time = time.time()
    # # 自回归生成未来30帧
    # with torch.no_grad():
    #     generated = clf(init_motion)
    #
    # print(f"运动分类形状: {generated.shape}")  # (30帧, 关节数, 3D坐标)
    # # 计算耗时
    # end_time = time.time()
    # print(f"生成耗时: {end_time - start_time:.4f}秒")
