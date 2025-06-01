# -*- coding: utf-8 -*-
"""Motion syllable (unsupervised action unit) segmentation pipeline
=================================================================
This file adds an **end‑to‑end pipeline** that plugs on top of the provided
`MotionVQVAE` model and produces variable‑length motion segments, each
assigned with an unsupervised discrete label.

The core steps follow the technical route we discussed earlier:

1. **Feature extraction** – use the RVQ code indices (optionally latent
   embeddings) returned by the VQ‑VAE as low‑dimensional descriptors and
   concatenate an energy feature ‖p_t − p_{t−1}‖.
2. **Unsupervised boundary detection** – compute a self‑similarity matrix
   (SSM) on the features, convolve it with a checkerboard kernel to obtain a
   novelty curve, then locate its peaks as candidate boundaries.
3. **Segment‑level representation & DP‑Means clustering** – aggregate each
   segment into a 65‑D vector and run Dirichlet‑Process‑Means to auto‑decide
   the number of clusters, producing the final motion‑syllable labels.

The whole procedure is wrapped in the `MotionSyllableSegmenter` class.

Example
-------
>>> vqvae = MotionVQVAE(...).eval().cuda()
>>> segmenter = MotionSyllableSegmenter(vqvae)
>>> rotmat, meta = vqvae(x=batch_x, valid=batch_valid)
>>> segments, labels = segmenter.segment(batch_x, batch_valid)
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import numpy as np

try:
    from scipy.signal import find_peaks
except ImportError:
    # Fallback peak detector – torch‑based simple argmax in sliding window
    def find_peaks(x, distance=1, prominence=0.0):  # type: ignore
        x = np.asarray(x)
        peaks = []
        for i in range(distance, len(x) - distance):
            if x[i] > prominence and x[i] == x[i - distance : i + distance + 1].max():
                peaks.append(i)
        return np.asarray(peaks), {}

###############################################################################
# Helper kernels / clustering utils                                           #
###############################################################################

def checkerboard_kernel(size: int = 5) -> Tensor:
    """Return a size×size Foote checkerboard kernel for novelty detection."""
    assert size % 2 == 1, "Kernel size must be odd."
    half = size // 2
    kernel = torch.ones(size, size)
    kernel[: half + 1, : half + 1] = -1
    kernel[: half + 1, half + 1 :] = 1
    kernel[half + 1 :, : half + 1] = 1
    kernel[half + 1 :, half + 1 :] = -1
    return kernel / (size * size)


def dp_means(x: np.ndarray, lam: float = 0.8, max_iter: int = 100) -> np.ndarray:
    """Simple Dirichlet‑Process‑Means clustering (isotropic, Euclidean).

    Parameters
    ----------
    x : (N, D) array
    lam : penalty lambda controlling cluster creation
    """
    N, D = x.shape
    # Initialise with first point
    centers = [x[0].copy()]
    labels = np.zeros(N, dtype=int)

    for _ in range(max_iter):
        # Assignment step
        dists = np.linalg.norm(x[:, None, :] - np.stack(centers)[None, :, :], axis=2)
        min_dist = dists.min(axis=1)
        labels = dists.argmin(axis=1)

        # Create new clusters where distance > lambda
        new_pts = x[min_dist > lam]
        if new_pts.size > 0:
            for p in new_pts:
                centers.append(p.copy())
            # Re‑compute assignments with new centers
            centers_np = np.stack(centers)
            dists = np.linalg.norm(x[:, None, :] - centers_np[None, :, :], axis=2)
            labels = dists.argmin(axis=1)

        # Update step
        new_centers = []
        for k in range(len(centers)):
            pts = x[labels == k]
            if len(pts) == 0:
                new_centers.append(centers[k])
            else:
                new_centers.append(pts.mean(axis=0))
        if np.allclose(np.stack(centers), np.stack(new_centers), atol=1e-4):
            break
        centers = new_centers
    return labels

###############################################################################
# Main segmentation class                                                     #
###############################################################################

class MotionSyllableSegmenter(nn.Module):
    """Plug‑and‑play unsupervised motion syllable (action unit) segmenter."""

    def __init__(
        self,
        vqvae: nn.Module,
        embed_dim: int = 64,
        kernel_size: int = 7,
        peak_prominence: float = 0.15,
        min_dist: int = 7,
        dp_lambda: float = 0.001,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.vqvae = vqvae.eval()  # freeze by default
        # vqvae 参数requires_grad=False
        for param in self.vqvae.parameters():
            param.requires_grad = False
        self.embed_dim = embed_dim
        self.register_buffer("checker", checkerboard_kernel(kernel_size)[None, None])
        self.peak_prominence = peak_prominence
        self.min_dist = min_dist
        self.dp_lambda = dp_lambda
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    @torch.inference_mode()
    def segment(
        self, x: Tensor, valid: Tensor
    ) -> Tuple[List[List[Tuple[int, int]]], List[List[int]]]:
        """Segment a **batch** of variable‑length motion sequences.

        Parameters
        ----------
        x : [B, T, V, C] – input motion (rotation matrices / joints etc.)
        valid : [B, T] – bool mask indicating valid frames (1) vs padding (0)

        Returns
        -------
        segments : list len B of lists of (start, end) tuples  (Python indices)
        labels   : list len B of lists of integer cluster labels (same length as segments[b])
        """
        x, valid = x.to(self.device), valid.to(self.device)
        feats = self._extract_features(x, valid)  # list of tensors per sample

        segments, labels = [], []
        for f, m in zip(feats, valid):  # per sample processing
            seg_bounds = self._detect_boundaries(f)  # list of ints (boundaries)
            seg_bounds = self._postprocess_bounds(seg_bounds, int(m.sum().item()))
            seg_feats = self._aggregate_segments(f, seg_bounds)
            seg_labels = dp_means(seg_feats, lam=self.dp_lambda)

            # convert boundaries to (start,end)
            seg_pairs = [(seg_bounds[i], seg_bounds[i + 1]) for i in range(len(seg_bounds) - 1)]
            segments.append(seg_pairs)
            labels.append(list(seg_labels))
        return segments, labels

    # ------------------------------------------------------------------
    # STEP 1 – feature extraction
    # ------------------------------------------------------------------

    def _extract_features(self, x: Tensor, valid: Tensor) -> List[Tensor]:
        """Return a list (len B) of [T_i, 65] tensors ready for SSM."""
        # Pass through VQ‑VAE encoder + quantizer – we only need indices / embeddings
        (rotmat,), _, indices = self.vqvae(x=x, valid=valid)
        # indices: [B, T, V, n_q]
        if indices.dim() == 4:
            # take first codebook level for efficiency
            indices = indices[..., 0]
        B, T, V = indices.shape

        # Build an embedding table on‑the‑fly (learned or sinusoid). For simplicity,
        # we map indices ∈ [0, n_e) to a fixed sinusoid embedding.
        n_e = self.vqvae.n_e
        freqs = torch.arange(self.embed_dim, device=x.device)[None, :]  # 1×D
        phases = 2 * math.pi * indices.reshape(-1, 1) / n_e  # (B*T*V)×1
        tok_emb = torch.sin(freqs * phases).view(B, T, V, self.embed_dim)

        # Energy feature: joint speed norm (mean over joints)
        pos = x[..., :3] if x.size(-1) >= 3 else rotmat[..., :3, 3]  # heuristic
        speed = (pos[:, 1:] - pos[:, :-1]).norm(dim=-1).mean(-1)  # [B,T-1]
        speed = torch.cat([speed[:, :1], speed], dim=1)  # pad first frame
        speed = speed.unsqueeze(-1)  # [B,T,1]

        feat = torch.cat([tok_emb.mean(2), speed], dim=-1)  # [B,T,65]

        feats = []
        for b in range(B):
            T_i = int(valid[b].sum().item())
            feats.append(feat[b, :T_i].detach())
        return feats

    # ------------------------------------------------------------------
    # STEP 2 – boundary detection
    # ------------------------------------------------------------------

    def _detect_boundaries(self, feat: Tensor) -> List[int]:
        """Foote novelty detection to obtain tentative boundaries.
        *Fix*: extract the **diagonal novelty curve** so the downstream
        `scipy.signal.find_peaks` receives a 1‑D array (the previous version
        mistakenly passed a 2‑D matrix).
        """
        T = feat.shape[0]
        # ---------------- self‑similarity ----------------
        S = torch.einsum("td,sd->ts", feat, feat)             # [T,T]
        pad = self.checker.shape[-1] // 2                     # same‑padding (PyTorch ≤1.12)
        S = F.pad(S[None, None], (pad, pad, pad, pad))        # (l,r,t,b)
        nov = F.conv2d(S, self.checker)[0, 0]                 # [T,T]

        # ---- take diagonal as 1‑D novelty curve ----
        nov_curve = nov.diagonal()                            # [T]
        nov_np = nov_curve.cpu().numpy()

        peaks, _ = find_peaks(
            nov_np,
            distance=self.min_dist,
            prominence=self.peak_prominence,
        )
        bounds = [0] + peaks.tolist() + [T]
        bounds = sorted(list(set(bounds)))
        if bounds[-1] != T:
            bounds.append(T)
        return bounds

    def _postprocess_bounds(self, bounds: List[int], T: int) -> List[int]:
        """Merge too‑short segments and clamp within [0,T]."""
        bounds = [b for b in bounds if 0 <= b <= T]
        bounds = sorted(list(set(bounds)))
        # merge segments shorter than 5 frames
        merged = [bounds[0]]
        for b in bounds[1:]:
            if b - merged[-1] < 5:
                continue
            merged.append(b)
        if merged[-1] != T:
            merged.append(T)
        return merged

    # ------------------------------------------------------------------
    # STEP 3 – segment aggregation & clustering
    # ------------------------------------------------------------------

    def _aggregate_segments(self, feat: Tensor, bounds: List[int]) -> np.ndarray:
        seg_vecs = []
        for s, e in zip(bounds[:-1], bounds[1:]):
            seg = feat[s:e]
            mean = seg.mean(0)
            std = seg.std(0)
            seg_vecs.append(torch.cat([mean, std]).cpu().numpy())
        return np.stack(seg_vecs)

###############################################################################
# Quick test (mock)                                                           #
###############################################################################

if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from motion_vqvae import MotionVQVAE
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
    seg = MotionSyllableSegmenter(vqvae, device=device)
    x = torch.randn(2, 120, 32, 6).to(device)
    valid = torch.ones(2, 120, dtype=torch.long).to(device)
    segs, labs = seg.segment(x, valid)
    for i in range(2):
        print(f"Sample {i}:", segs[i], labs[i])


    # 计算模型参数量
    num_params = sum(p.numel() for p in seg.parameters() if p.requires_grad)
    print(f"Number of parameters in the model: {num_params:}")
