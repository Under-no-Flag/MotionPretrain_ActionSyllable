# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
from typing import List

import torch
import torch.nn as nn
from typing import List, Sequence
from torch.functional import F

""" VectorQuantizer code adapted from by https://github.com/CompVis/taming-transformers/taming/modules/vqvae/quantize.py"""

__all__ = ['VectorQuantizer']


def L2_efficient(x, y):
    return (x.pow(2).sum(1, keepdim=True) - 2 * x @ y + y.pow(2).sum(0, keepdim=True))


class EmaCodebookMeter:
    """Compute an estimate of centroid usage, using an EMA to track proportions """

    def __init__(self, codebook_size, ema_alpha=0.05):
        self.codebook_size = codebook_size
        self.bins = (torch.ones((self.codebook_size), requires_grad=False) / self.codebook_size).detach().cuda()
        self.ema_alpha = ema_alpha
        self.iters = 0

    def bincount(self, val, weights=None):
        norm = val.shape[0]
        weights = weights.reshape(-1) if weights is not None else None
        count = torch.bincount(val.reshape(-1), minlength=self.codebook_size,
                               weights=weights).detach()
        self.iters += 1
        return count / norm

    def load(self, bins):
        self.bins = torch.tensor(bins, requires_grad=False).detach().cuda()

    def update(self, val, weights=None, n=1):
        """ Count usage of each value in the codebook """
        count = self.bincount(val, weights=weights)
        alpha = max(self.ema_alpha, 1 / (self.iters + 1))
        self.bins = (1. - alpha) * self.bins + alpha * count

    def get_hist(self):
        return self.bins


class VectorQuantizer(nn.Module):
    """
    Code taken from https://github.com/CompVis/taming-transformers/
            blob/9d17ea64b820f7633ea6b8823e1f78729447cb57/taming/
            modules/vqvae/quantize.py#L213
    for handling input of shape [batch_size, seq_len, hid_dim]

    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """

    def __init__(self, n_e, e_dim, beta,
                 nbooks=1, balance=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.nbooks = nbooks
        self.balance = balance

        assert n_e % nbooks == 0, "nb codebooks should divide nb centroids"
        self.n_e_i = n_e // nbooks

        embed_dims = (nbooks - 1) * [e_dim // nbooks] + \
                     [e_dim - (nbooks - 1) * (e_dim // nbooks)]
        self.embed_dims = embed_dims

        self.embeddings = torch.nn.ModuleDict({str(i): nn.Embedding(self.n_e_i, d) for i, d in enumerate(embed_dims)})

        self.trackers = {}
        for i, e in self.embeddings.items():
            e.weight.data.uniform_(-1.0 / self.n_e_i, 1.0 / self.n_e_i)
            print(f"Codebook {i}: {list(e.weight.size())}")

            self.trackers[int(i)] = EmaCodebookMeter(self.n_e_i)

    def get_state(self):
        return {i: self.trackers[i].get_hist().cpu().data.numpy() for i in self.trackers.keys()}

    def load_state(self, bins):
        for i, b in bins.items():
            self.trackers[i].load(b)

    def get_hist(self, i):
        return self.trackers[i].get_hist()

    def reset(self, i):
        for i in self.trackers.keys():
            self.trackers = EmaCodebookMeter(self.embed_dims[int(i)])

    def track_assigment(self, emb_ind, i):
        self.trackers[i].update(emb_ind)

    def forward_one(self, z, i, weights=None):
        bsize = self.e_dim // self.nbooks
        e_dim = bsize if i < self.nbooks - 1 else self.e_dim - (self.nbooks - 1) * bsize

        z_flattened = z.view(-1, e_dim)
        dist = L2_efficient(z_flattened, self.embeddings[str(i)].weight.t())

        if self.balance and weights is not None:
            # weights = (proportions * self.n_embed).unsqueeze(0)
            wdist = dist * weights.unsqueeze(0)
            dist = -torch.nn.functional.softmax(-wdist, 1)

        min_encoding_indices = torch.argmin(dist, dim=1).unsqueeze(1)
        min_encodings = torch.zeros(min_encoding_indices.shape[0], self.n_e_i).to(z)
        min_encodings.scatter_(1, min_encoding_indices, 1)

        if self.training:
            self.track_assigment(min_encoding_indices.detach(), i)

        # get quantized latent vectors
        z_q = torch.matmul(min_encodings, self.embeddings[str(i)].weight).view(z.shape)

        # min_encoding_indices.view(z.shape)
        return z_q, min_encoding_indices.view(z.shape[:-1] + (1,))

    def forward(self, z, p=1.0):
        assert z.size(2) == self.e_dim
        zs = torch.split(z, z.size(2) // len(self.embeddings), dim=-1)
        zq_i = [self.forward_one(z, i, self.get_hist(i)) for i, z in enumerate(zs)]
        z_q, min_encoding_indices = [torch.cat([e[i] for e in zq_i], dim=-1) for i in [0, 1]]

        # compute loss for embedding
        loss = torch.mean((z_q.detach() - z) ** 2, dim=-1) + self.beta * \
               torch.mean((z_q - z.detach()) ** 2, dim=-1)

        if p != 1.0:
            # Proba of being quantized.
            quant_mask = torch.bernoulli(p * torch.ones_like(z)).float()
            z_q = quant_mask * z_q + (1 - quant_mask) * z

        # preserve gradients
        z_q = z + (z_q - z).detach()
        return z_q, loss, min_encoding_indices

    def get_codebook_entry(self, indices, eos_mask=None):
        """
        Args:
            - indices: [batch_size,seq_len]
        Return:
            - z_q: [batch_size,seq_len,e_dim]
        """
        # This is a hack, but it enables us to keep the '-1' index solely in the gpt
        embds = [self.embeddings[str(i)](e.squeeze(-1)) for i, e in enumerate(torch.split(indices, 1, dim=-1))]
        return torch.cat(embds, dim=-1)


# motion chain quantizer
# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

import torch
import torch.nn as nn


class MCVectorQuantizer(nn.Module):
    """Vector‑Quantizer with *single‑pass* per joint.

    重复量化同一关节会导致梯度冲突和 codebook 统计失真。这里通过
    `visited` 标记确保每个关节仅量化一次；如果同一关节在后续 motion
    chain 中再次出现，则直接复用第一次得到的量化结果与嵌入，而不再产生
    新的梯度或修改 tracker。这样可以保证梯度方向一致、收敛更稳定。
    """

    def __init__(self,
                 num_joints: int = 32,
                 n_e: int = 1024,
                 e_dim: int = 128,
                 beta: float = 0.25,
                 nbooks: int = 11,  # unused but kept for compatibility
                 balance: bool = False,
                 mlp_hidden: int = 256):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.num_joints = num_joints

        # === codebooks =====================================================
        self.embeddings = nn.ModuleDict({
            str(i): nn.Embedding(n_e, e_dim) for i in range(num_joints)
        })
        for e in self.embeddings.values():
            nn.init.xavier_uniform_(e.weight)

        # === motion chains ==================================================
        self.motion_chain: List[List[int]] = [
            [0, 1, 2, 3, 4, 5],
            [0, 6, 7, 8, 9, 10],
            [0, 11, 12, 13, 14, 15],
            [12, 16, 17, 18, 19, 20, 21, 22, 23],
            [12, 24, 25, 26, 27, 28, 29, 30, 31],
        ]

        # === usage trackers =================================================
        self.trackers = {i: EmaCodebookMeter(n_e) for i in range(num_joints)}

        # === parent‑conditioned MLP ========================================
        self.mlp = nn.Sequential(
            nn.Linear(e_dim * 2, mlp_hidden),
            nn.LayerNorm(mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, e_dim),
        )

    # ---------------------------------------------------------------------
    # helper ----------------------------------------------------------------
    def _quantize_joint(self, z, joint_idx, p=1.0):
        """Quantize single joint; returns (z_q, e_joint, idx_joint, loss)."""
        B, T, C = z.shape
        z_flat = z.reshape(-1, C)

        dist = torch.cdist(z_flat, self.embeddings[str(joint_idx)].weight)
        idx = torch.argmin(dist, dim=1)
        one_hot = torch.zeros_like(dist).scatter_(1, idx.unsqueeze(1), 1)

        e_joint = (one_hot @ self.embeddings[str(joint_idx)].weight).view(B, T, C)



        if p < 1.0:
            mask = torch.bernoulli(p * torch.ones(B, T, 1, device=z.device))
            z_q = mask * e_joint + (1 - mask) * z


        commit = (e_joint.detach() - z).pow(2).mean()
        codebook = (e_joint - z.detach()).pow(2).mean()
        loss = commit + self.beta * codebook

        self.trackers[joint_idx].update(idx.detach())

        #  针对ST trick 的修改
        z_q = z + (e_joint - z).detach()
        return z_q, e_joint, idx.view(B, T), loss

    # ---------------------------------------------------------------------
    def forward(self, z, p: float = 1.0):
        """Quantize the entire motion sequence with single‑pass per joint."""
        B, T, V, C = z.shape
        device = z.device

        z_q = torch.zeros_like(z)
        indices = torch.full((B, T, V), -1, dtype=torch.long, device=device)
        total_loss = 0.0

        # bookkeeping for already‑processed joints
        visited = [False] * V
        stored_q, stored_e, stored_idx = [None] * V, [None] * V, [None] * V

        for chain in self.motion_chain:
            parent_e = None
            for i, joint_idx in enumerate(chain):
                # — reuse if this joint was already processed —
                if visited[joint_idx]:
                    q_joint = stored_q[joint_idx]
                    e_joint = stored_e[joint_idx]
                    idx_joint = stored_idx[joint_idx]
                else:
                    z_joint = z[:, :, joint_idx, :]
                    h_joint = (
                        self.mlp(torch.cat([parent_e, z_joint], -1)) if i > 0 else z_joint
                    )
                    q_joint, e_joint, idx_joint, loss = self._quantize_joint(
                        h_joint, joint_idx, p
                    )

                    # cache & bookkeep
                    visited[joint_idx] = True
                    stored_q[joint_idx] = q_joint
                    stored_e[joint_idx] = e_joint
                    stored_idx[joint_idx] = idx_joint
                    total_loss += loss

                # write outputs (always – even when reused)
                z_q[:, :, joint_idx, :] = q_joint
                indices[:, :, joint_idx] = idx_joint
                parent_e = e_joint.detach()

        # divisor = #unique joints actually quantized (== sum(visited))
        total_loss = total_loss / max(sum(visited), 1)
        return z_q, total_loss, indices

    # ---------------------------------------------------------------------
    def get_codebook_entry(self, indices):
        B, T, V = indices.shape
        z_q = torch.zeros(B, T, V, self.e_dim, device=indices.device)
        for v in range(V):
            z_q[:, :, v] = self.embeddings[str(v)](indices[:, :, v])
        return z_q

class ResidualVectorQuantizer(nn.Module):
    """Residual Vector Quantizer (RVQ).

    RVQ applies *n_q* successive, independent codebooks.  At stage *k*, the
    residual error from the previous stage is quantised and **added** to the
    running reconstruction.  Compared with a single huge code‑book, RVQ can
    achieve finer approximation with fewer parameters and better usage of the
    embedding space.

    The class follows the same interface as :class:`VectorQuantizer` so it can
    be used as a drop‑in replacement, e.g. inside *MotionVQVAE*.
    """

    def __init__(
        self,
        *,
        n_q: int = 4,
        n_e: int = 1024,
        e_dim: int = 128,
        beta: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_q = n_q
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta

        # create independent codebooks for each stage -----------------
        self.embeddings = nn.ModuleDict({str(i): nn.Embedding(n_e, e_dim) for i in range(n_q)})
        for emb in self.embeddings.values():
            nn.init.xavier_uniform_(emb.weight)

        # usage trackers (optional) -----------------------------------
        self.trackers = {i: EmaCodebookMeter(n_e) for i in range(n_q)}

    # ------------------------------------------------------------------
    def _quantize_stage(self, residual: torch.Tensor, stage: int):
        """Quantise *residual* using code‑book #*stage* (no ST trick here)."""
        flat = residual.view(-1, self.e_dim)
        dist = L2_efficient(flat, self.embeddings[str(stage)].weight.t())
        idx = torch.argmin(dist, dim=1)
        one_hot = torch.zeros_like(dist).scatter_(1, idx.unsqueeze(1), 1)
        q = (one_hot @ self.embeddings[str(stage)].weight).view_as(residual)

        if self.training:
            self.trackers[stage].update(idx.detach())

        return q, idx.view(*residual.shape[:-1], 1)

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor, p: float = 1.0):
        """Quantise *z* with RVQ; returns (z_q, loss, indices)."""
        assert z.shape[-1] == self.e_dim, "last dim must equal e_dim"

        residual = z
        cumulative_q = 0.0
        losses = []
        indices = []

        for k in range(self.n_q):
            q_k, idx_k = self._quantize_stage(residual, k)

            # optional stochastic drop (same semantics as VectorQuantizer)
            if p != 1.0:
                mask = torch.bernoulli(p * torch.ones_like(q_k))
                q_k = mask * q_k + (1 - mask) * residual

            # --- commitment / codebook losses ----------------------
            commit = (residual.detach() - q_k).pow(2).mean(-1)
            codebook = (q_k - residual.detach()).pow(2).mean(-1)
            losses.append(commit + self.beta * codebook)

            cumulative_q = cumulative_q + q_k
            residual = residual - q_k  # update residual
            indices.append(idx_k)

        z_q = cumulative_q
        # straight‑through estimator – gradients through q only ---------
        z_q = z + (z_q - z).detach()

        total_loss = torch.stack(losses).mean(0)  # average across stages
        indices = torch.cat(indices, dim=-1)
        return z_q, total_loss.mean(), indices

    # ------------------------------------------------------------------
    def get_codebook_entry(self, indices: torch.Tensor):
        """Decode *indices* back to embeddings and **sum** over stages."""
        parts = torch.split(indices, 1, dim=-1)
        embs = [self.embeddings[str(i)](p.squeeze(-1)) for i, p in enumerate(parts)]
        return torch.stack(embs, dim=0).sum(0)

class ResidualVectorQuantizerEMA(ResidualVectorQuantizer):
    def __init__(self, *args, decay=0.99, eps=1e-5, **kw):
        super().__init__(*args, **kw)
        self.decay, self.eps = decay, eps
        # 为每个 codebook加 EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(self.n_q, self.n_e))
        self.register_buffer("ema_embed",        torch.zeros(self.n_q, self.n_e, self.e_dim))

    def _quantize_stage(self, residual, stage):
        flat = residual.view(-1, self.e_dim)
        emb  = self.embeddings[str(stage)].weight      # (n_e, e_dim)
        dist = L2_efficient(flat, emb.t())
        idx  = torch.argmin(dist, dim=1)
        one_hot = torch.zeros_like(dist).scatter_(1, idx.unsqueeze(1), 1)

        # ---------- EMA UPDATE ----------
        if self.training:
            with torch.no_grad():  # ---①
                n = one_hot.sum(0)
                embed_sum = one_hot.T @ flat

                # 下面统统用 .mul_ / .add_ 做 **原位更新**，避免重新分配
                self.ema_cluster_size[stage].mul_(self.decay).add_(n, alpha=1 - self.decay)
                self.ema_embed[stage].mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

                # 归一化
                den = self.ema_cluster_size[stage] + self.eps
                self.embeddings[str(stage)].weight.copy_(self.ema_embed[stage] / den.unsqueeze(1))

            # 同时仍可用 tracker 记录 usage 分布
            self.trackers[stage].update(idx.detach())

        q = emb[idx].view_as(residual)
        return q, idx.view(*residual.shape[:-1], 1)


class ResidualVectorQuantizerGCN(nn.Module):
    def __init__(
            self,
            *,
            adjacency: torch.Tensor,  # V×V 的邻接矩阵
            e_dim: int,
            **rvq_kwargs,  # 其余传给 ResidualVectorQuantizerGCN 基类，如 n_q, n_e, beta
    ):
        super().__init__()
        # 复用之前的 RVQ 初始化
        self.rvq = ResidualVectorQuantizer(e_dim=e_dim, **rvq_kwargs)

        V = adjacency.size(0)
        self.V = V
        # 对称归一化邻接
        A = adjacency + torch.eye(V, device=adjacency.device)
        D = A.sum(1)
        D_inv_sqrt = torch.diag(D.pow(-0.5))
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
        self.register_buffer("A_norm", A_norm)  # V×V

        # 一个简单的线性层做消息变换
        self.gcn_lin = nn.Linear(e_dim, e_dim)
        self.relu = nn.LeakyReLU(0.2)
        self.ln = nn.LayerNorm(e_dim) # 如果 e_dim 是特征维度
        self.alpha = 0.1

    def _gcn_refine(self, residual: torch.Tensor, prev_emb: torch.Tensor = None):
        """
        residual: [N, V, d]
        prev_emb:  [N, V, d] or None
        返回 [N, V, d]
        """
        # 融合前 k-1 级量化态
        # fused = residual if prev_emb is None else residual + prev_emb
        fused = residual
        # GCN 聚合
        neigh = torch.einsum("vw,nwd->nvd", self.A_norm, fused)
        neigh = self.gcn_lin(neigh)
        neigh = self.relu(neigh)  # [N, V, d]
        neigh = self.ln(neigh) # LayerNorm 直接作用于最后一个维度
        return fused + self.alpha*neigh  # [N, V, d]

        # return residual

    def forward(self, z: torch.Tensor, p: float = 1.0):
        """
        z: [B, T, V, C]
        返回:
          z_q:      [B, T, V, C]
          loss:     scalar
          indices:  [B, T, V, n_q]
        """
        B, T, V, d = z.shape
        assert V == self.V and d == self.rvq.e_dim

        # 合并 B,T 方便批量处理
        z_flat = z.view(B * T, V, d)  # [N, V, d], N=B*T

        residual = z_flat
        cumulative_q = torch.zeros_like(z_flat)
        losses = []
        indices = []
        prev_idx = None

        # 按级量化
        for k in range(self.rvq.n_q):
            # 如果有前 k 级 idx，就解码出前 k 级 embedding
            prev_emb = (
                self.rvq.get_codebook_entry(prev_idx)
                if prev_idx is not None else None
            )  # [N, V, d] or None

            # 1) 用 GCN 融合 refine
            refined = self._gcn_refine(residual, prev_emb)

            # 2) 再用RVQ原来的 _quantize_stage（只不过我们手动传入 refined）
            #    这里绕过了原class的 residual 参数，直接把flat和stage送进去
            emb_weight = self.rvq.embeddings[str(k)].weight  # (n_e, d)
            flat = refined.view(-1, d)  # [N*V, d]
            dist = L2_efficient(flat, emb_weight.t())  # [NV, n_e]
            idx = torch.argmin(dist, dim=1)  # [NV]
            one_hot = torch.zeros_like(dist).scatter_(1, idx.unsqueeze(1), 1)
            q_flat = one_hot @ emb_weight  # [NV, d]
            q = q_flat.view(B * T, V, d)  # [N, V, d]

            if self.training:
                self.rvq.trackers[k].update(idx.detach())

            # 随机丢弃（保持原 semantics）
            if p != 1.0:
                mask = torch.bernoulli(p * torch.ones_like(q))
                q = mask * q + (1 - mask) * residual

            # 损失
            commit = (residual.detach() - q).pow(2).mean(-1)  # [N, V]
            codebok = (q - residual.detach()).pow(2).mean(-1)  # [N, V]
            losses.append(commit + self.rvq.beta * codebok)

            cumulative_q = cumulative_q + q
            residual = residual - q

            idx = idx.view(B * T, V, 1)  # [N, V, 1]
            indices.append(idx)
            prev_idx = torch.cat(indices, dim=-1)  # [N, V, k+1]

        # STE
        zq_flat = z_flat + (cumulative_q - z_flat).detach()  # [N, V, d]
        total_loss = torch.stack(losses, 0).mean()

        # 恢复形状
        z_q = zq_flat.view(B, T, V, d)
        indices = prev_idx.view(B, T, V, self.rvq.n_q)  # [B, T, V, n_q]

        return z_q, total_loss, indices

class ResidualVectorQuantizerGCNEMA(ResidualVectorQuantizerGCN):
    """
    GCN-conditioned RVQ + 指数移动平均(EMA) 码本更新。
    继承自 ResidualVectorQuantizerGCN，只改动 _quantize-and-update 的
    部分，其余接口保持不变。
    """
    def __init__(
        self,
        *,
        adjacency: torch.Tensor,
        e_dim: int,
        decay: float = 0.95,
        eps: float = 1e-5,
        **rvq_kwargs,
    ):
        super().__init__(adjacency=adjacency, e_dim=e_dim, **rvq_kwargs)

        # === 为每个 code-book 配置 EMA 缓存 ===========================
        self.decay, self.eps = decay, eps
        self.register_buffer("ema_cluster_size",
                             torch.zeros(self.rvq.n_q, self.rvq.n_e))
        self.register_buffer("ema_embed",
                             torch.zeros(self.rvq.n_q, self.rvq.n_e, self.rvq.e_dim))

    # ------------------------------------------------------------------
    def _quantize_stage_with_ema(
        self,
        refined: torch.Tensor,          # [N, V, d] —— GCN 处理后的特征
        stage: int
    ):
        """检索最近邻 **并** 执行 EMA 更新。返回 (q, idx, one_hot)。"""
        N, V, d = refined.shape
        flat = refined.reshape(-1, d)                     # [N*V, d]
        emb_weight = self.rvq.embeddings[str(stage)].weight  # (n_e, d)
        dist = L2_efficient(flat, emb_weight.t())         # [NV, n_e]
        idx  = torch.argmin(dist, dim=1)                  # [NV]
        one_hot = torch.zeros_like(dist).scatter_(1, idx.unsqueeze(1), 1)

        # -------------------- EMA 更新 ------------------------------
        if self.training:
            with torch.no_grad():
                n = one_hot.sum(0)                        # [n_e]
                embed_sum = one_hot.T @ flat              # [n_e, d]

                self.ema_cluster_size[stage].mul_(self.decay).add_(n,
                                       alpha=1 - self.decay)
                self.ema_embed[stage].mul_(self.decay).add_(embed_sum,
                                       alpha=1 - self.decay)

                # 归一化得到新的码本向量
                denom = (self.ema_cluster_size[stage] + self.eps).unsqueeze(1)
                self.rvq.embeddings[str(stage)].weight.copy_(
                    self.ema_embed[stage] / denom
                )

            # 继续记录 tracker（可选）
            self.rvq.trackers[stage].update(idx.detach())

        # 取回量化向量
        q_flat = emb_weight[idx]                          # [NV, d]
        q = q_flat.view(N, V, d)
        return q, idx.view(N, V, 1)                       # idx 形状保持一致

    # ------------------------------------------------------------------
    def forward(self, z: torch.Tensor, p: float = 1.0):
        """
        与父类接口完全一致，只把单级量化逻辑换成 _quantize_stage_with_ema。
        """
        B, T, V, d = z.shape
        assert V == self.V and d == self.rvq.e_dim

        # ---- 展平 B,T -------------------------------------------------
        z_flat = z.view(B * T, V, d)          # N = B*T
        residual     = z_flat
        cumulative_q = torch.zeros_like(z_flat)
        losses, indices = [], []
        prev_idx = None

        for k in range(self.rvq.n_q):
            # === 1. 取前 k-1 级嵌入并做 GCN refine ====================
            prev_emb = (self.rvq.get_codebook_entry(prev_idx)
                        if prev_idx is not None else None)
            refined = self._gcn_refine(residual, prev_emb)   # [N,V,d]

            # === 2. 最近邻 + EMA 更新 =================================
            q_k, idx_k = self._quantize_stage_with_ema(refined, k)

            # === 3. 随机丢弃（保持原语义） ============================
            if p != 1.0:
                mask = torch.bernoulli(p * torch.ones_like(q_k))
                q_k  = mask * q_k + (1 - mask) * residual

            # === 4. 计算损失、累计结果 ================================
            commit = (refined.detach() - q_k).pow(2).mean(-1)
            codebok = (q_k - refined.detach()).pow(2).mean(-1)
            losses.append(commit + self.rvq.beta * codebok)

            cumulative_q += q_k
            residual     -= q_k

            indices.append(idx_k)
            prev_idx = torch.cat(indices, dim=-1)           # [N, V, k+1]

        # ---- 直通估计（STE） -----------------------------------------
        zq_flat = z_flat + (cumulative_q - z_flat).detach()
        total_loss = torch.stack(losses, 0).mean()

        # ---- 恢复形状并返回 ------------------------------------------
        z_q     = zq_flat.view(B, T, V, d)
        indices = prev_idx.view(B, T, V, self.rvq.n_q)
        return z_q, total_loss, indices


if __name__ == "__main__":
    B, T, V, C = 4, 64, 32, 64
    x = torch.randn(B, T, V, C, device="cuda")
    device = torch.device('cuda')

    A = torch.zeros(32, 32)
    edges=[(0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4), (4, 5), (5, 6)]
    for i, j in edges:
        A[i, j] = A[j, i] = 1
    model= ResidualVectorQuantizerGCNEMA(
        n_q=6, n_e=256, e_dim=C,adjacency=A,
    ).to(device)
    z_q, loss, idx = model(x)
    print(z_q.shape, loss.item(), idx.shape)

    model = MCVectorQuantizer(n_e=1024, e_dim=C, beta=0.25, nbooks=32, balance=False).to(device)
    rvq = ResidualVectorQuantizerEMA(n_q=8, n_e=512, e_dim=C).cuda()
    z_q, loss, idx = rvq(x)
    print("RVQ", z_q.shape, loss.item(), idx.shape)
    # two forwards
    z_q1, _, indices1 = model(x)
    z_q2, _, indices2 = model(x)
    print((z_q1 - z_q2).abs().sum())
    print(z_q1.shape, indices1.shape)

