# model/blocks/down_up_sample.py
import torch
from torch import nn
from einops import rearrange, repeat

class SpatioTemporalDown(nn.Module):
    """
    (B, T, V, C) -> (B, T//s_t, V//s_v, C')
    时域用一维深度可分离卷积(可保持边界)，
    关节维用分组线性投影(保证拓扑相关性)。
    """
    def __init__(self, c_in, c_out, stride_t=2, stride_v=2, groups=4):
        super().__init__()
        self.s_t, self.s_v = stride_t, stride_v
        # self.time_conv = nn.Conv1d(
        #     c_in, c_out, kernel_size=3, stride=stride_t,
        #     padding=1, groups=groups, bias=False)
        self.time_conv = nn.Sequential(
            nn.Conv1d(c_in, c_out, 3, stride_t, padding=1, groups=groups, bias=False),
            nn.BatchNorm1d(c_out),  # 稳定训练
            nn.GELU()
        )

        self.joint_pool = nn.Linear(stride_v, 1, bias=False)  # 每 stride_v 个关节聚合为 1

    def forward(self, x):  # x: (B,T,V,C)
        # B, T, V, C = x.shape
        # # ---- 1. time ↓ ------------------------------------------------
        # xt = rearrange(x, 'b t v c -> (b v) c t')
        # xt = self.time_conv(xt)  # (b·v, c_out, T//s_t)
        # xt = rearrange(xt, '(b v) c t -> b t v c', b=B, v=V)
        # # ---- 2. joint ↓ -----------------------------------------------
        # v_new = V // self.s_v
        # # 变成 (B,T,V',C,s_v)  —— 让 **s_v 在最后一维**
        # x_blocks = rearrange(
        #     xt, 'b t (v g) c -> b t v c g', g=self.s_v)  # g = stride_v
        # # 线性层只作用在最后一维 g
        # xj = self.joint_pool(x_blocks).squeeze(-1)  # (B,T,V',C)
        # return xj
        # # (B,T',V',C')

        B, T, V, C = x.shape
        # ---- 1. time ↓ ------------------------------------------------
        xt = rearrange(x, 'b t v c -> (b v) c t')
        xt = self.time_conv(xt)  # (b·v, c_out, T//s_t)
        xt = rearrange(xt, '(b v) c t -> b t v c', b=B, v=V)

        return xt

from einops import rearrange
import torch.nn as nn

class SpatioTemporalUp(nn.Module):
    def __init__(self, c_in, c_out, stride_t=2, stride_v=2):
        super().__init__()
        self.s_t, self.s_v = stride_t, stride_v
        # self.time_up = nn.ConvTranspose1d(
        #     c_in, c_out,
        #     kernel_size=stride_t * 2,      # (=4 when stride=2)
        #     stride=stride_t,               # (=2)
        #     padding=stride_t - 1)          # (=1)  ↔ Down 的 kernel=3,pad=0

        self.time_up = nn.Sequential(
            nn.ConvTranspose1d(c_in, c_out, stride_t * 2, stride_t, padding=stride_t - 1, bias=False),
            nn.BatchNorm1d(c_out),
            nn.GELU()
        )
        self.joint_up = nn.Linear(1, stride_v, bias=False)

    def forward(self, x, T_out, V_out):
        """
        x     : (B, T', V', C)
        T_out : 目标时间长度 (= T' * stride_t)
        V_out : 目标关节数   (= V' * stride_v)
        """
        # B, T_, V_, C = x.shape                    # B·V' 之后会被合批
        #
        # # ---- 1. joint ↑ -------------------------------------------------
        # xj = self.joint_up(x.unsqueeze(-1))       # (B,T',V',C,s_v)
        # xj = (xj.permute(0, 1, 2, 4, 3)           # (B,T',V',s_v,C)
        #          .contiguous()
        #          .view(B, T_, V_out, C))          # (B,T',V_out,C)

        # ---- 2. time ↑ --------------------------------------------------
        # xt = rearrange(xj, 'b t v c -> (b v) c t')          # (B·V_out, C, T')
        # out_size = (xt.shape[0], self.time_up[0].out_channels, T_out)
        # xt = self.time_up(xt)         # (B·V_out, C, T_out)
        # xt = rearrange(xt, '(b v) c t -> b t v c', b=B, v=V_out)
        # return xt                                           # (B, T_out, V_out, C)

        B, T_, V_, C = x.shape
        xt = rearrange(x, 'b t v c -> (b v) c t')          # (B·V_out, C, T')
        out_size = (xt.shape[0], self.time_up[0].out_channels, T_out)
        xt = self.time_up(xt)         # (B·V_out, C, T_out)
        xt = rearrange(xt, '(b v) c t -> b t v c', b=B, v=V_out)
        return xt                                           # (B, T_out, V_out, C)

# model/blocks/ga_pool.py
import torch
from torch import nn
from einops import rearrange

class GAPool(nn.Module):
    """
    Graph-Attention Pooling
    (B,T,V,C) -> (B,T,P,C)  ，P << V
    """
    def __init__(self, V, P, C, heads=4, dropout=0.0):
        super().__init__()
        self.P, self.h = P, heads
        d = C // heads

        # 投影为 Q, K, V'
        self.q_proj = nn.Linear(C, C, bias=False)
        self.k_proj = nn.Linear(C, C, bias=False)
        self.v_proj = nn.Linear(C, C, bias=False)
        # learnable part query：  (heads, P, d)
        self.part_query = nn.Parameter(torch.randn(heads, P, d))
        nn.init.xavier_uniform_(self.part_query)

        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(C, C, bias=False)

    def forward(self, x, adj=None):
        """
        x   : (B,T,V,C)
        adj : (V,V) or None   — 可选硬骨骼 mask，值 0/-inf
        """
        B,T,V,C = x.shape
        h,d = self.h, C // self.h

        # ---- 0. 投影 -------------------------------------------------
        q = self.part_query                         # (h,P,d)
        k = rearrange(self.k_proj(x), 'b t v (h d) -> b h t v d', h=h)
        v = rearrange(self.v_proj(x), 'b t v (h d) -> b h t v d', h=h)

        # ---- 1. 注意力得分 ------------------------------------------
        #   α = softmax( q·kᵀ / √d + mask )
        attn = torch.einsum('hpd, b h t v d -> b h t p v', q, k) / d**0.5   # (B,h,T,P,V)
        if adj is not None:
            attn = attn + adj.view(1,1,1,1,V)      # 使非法连接=-inf
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # ---- 2. 聚合 -------------------------------------------------
        out = torch.einsum('b h t p v, b h t v d -> b h t p d', attn, v)    # (B,h,T,P,d)
        out = rearrange(out, 'b h t p d -> b t p (h d)')
        out = self.out_proj(out)                   # (B,T,P,C)
        return out, attn                           # attn 可视化 / 上采样用

class TimeDWConv(nn.Module):
    """(B,T,P,C) -> (B,T//s_t,P,C)"""
    def __init__(self, C, stride=2):
        super().__init__()


        self.conv = nn.Sequential(
            nn.Conv1d(C, C, kernel_size=stride*2, stride=stride, padding=1, groups=C, bias=False),
            nn.BatchNorm1d(C),  # 如果 batch 很小可换 GroupNorm(32, C)
            nn.SiLU()  # 比 ReLU 平滑
        )

    def forward(self, x):
        b,t,p,c= x.shape
        x = rearrange(x, 'b t p c -> (b p) c t')
        x = self.conv(x)
        x = rearrange(x, '(b p) c t -> b t p c', p=p)    # p 未变
        return x

class GAUnpool(nn.Module):
    """
    反向注意力：用保存的 αᵀ 把 part-tokens 撒回 V 关节
    输入  (B,T',P,C) + α:(B,h,T',P,V')  ->  (B,T',V',C)
    """
    def __init__(self, C, heads=4):
        super().__init__()
        self.h, self.C = heads, C
        self.out_proj = nn.Linear(C, C, bias=False)
        self.norm = nn.LayerNorm(C)

    def forward(self, part_x, attn):
        """
        part_x : (B,T',P,C)
        attn   : (B,h,T',P,V') —— 来自对应 GAPool
        """
        B,Tp,P,C = part_x.shape
        h,d = self.h, C // self.h

        v = rearrange(part_x, 'b t p (h d) -> b h t p d', h=h)
        # 逆聚合： V ← P
        full = torch.einsum('b h t p v, b h t p d -> b h t v d',
                            attn, v)                # (B,h,T,V',d)
        full = rearrange(full, 'b h t v d -> b t v (h d)')
        full = self.out_proj(full)
        full= self.norm(full)            # (B,T,V,C)
        return full

class GA_Down(nn.Module):
    """(B,T,V,C) → (B,T↓,P,C)"""
    def __init__(self, V, P, C, stride_t=2, heads=4):
        super().__init__()
        self.gapool = GAPool(V,P,C,heads)
        self.t_pool = TimeDWConv(C,stride_t)

    def forward(self, x, adj=None):
        x,atten = self.gapool(x,adj)        # 空间 ↓
        x   = self.t_pool(x)            # 时间 ↓
        return x,atten

class GA_Up(nn.Module):
    """(B,T',P,C) → (B,T,V,C)"""
    def __init__(self, V, P, C, stride_t=2, heads=4):
        super().__init__()
        self.t_up   = nn.ConvTranspose1d(C,C,
                        kernel_size=stride_t*2, stride=stride_t, padding=1)
        self.gaun   = GAUnpool(C,heads)
        self.V = V

    def forward(self, x_part, attn_cache, T_out):
        # ----- 时间 ↑ -------------------------
        x = rearrange(x_part, 'b t p c -> (b p) c t')
        x = self.t_up(x, output_size=(x.shape[0], self.t_up.out_channels, T_out))
        x = rearrange(x, '(b p) c t -> b t p c', p=x_part.shape[2])

        # ----- 空间 ↑ ------------------------
        full = self.gaun(x, attn_cache)           # (B,T,V,C)
        return full
