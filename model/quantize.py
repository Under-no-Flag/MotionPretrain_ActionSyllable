# Copyright (C) 2022-2023 Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
from typing import List

import torch
import torch.nn as nn
from typing import List, Sequence


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



if __name__ == "__main__":
    B, T, V, C = 4, 64, 32, 64
    x = torch.randn(B, T, V, C, device="cuda")
    device = torch.device('cuda')





    model = MCVectorQuantizer(n_e=1024, e_dim=C, beta=0.25, nbooks=32, balance=False).to(device)
    rvq = ResidualVectorQuantizer(n_q=8, n_e=512, e_dim=C).cuda()
    z_q, loss, idx = rvq(x)
    print("RVQ", z_q.shape, loss.item(), idx.shape)
    # two forwards
    z_q1, _, indices1 = model(x)
    z_q2, _, indices2 = model(x)
    print((z_q1 - z_q2).abs().sum())
    print(z_q1.shape, indices1.shape)

