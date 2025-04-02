import torch
import torch.nn as nn
import math
from einops import rearrange

""" Slightly adapted from  https://github.com/karpathy/minGPT/blob/master/mingpt/model.py """

class CausalSelfAttention(nn.Module):
    """
    Self-attention, possibly causal.
    """

    def __init__(self, config, in_dim=None):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.causal = config.causal
        # key, query, value projections for all heads
        if in_dim is None:
            in_dim = config.n_embd
        self.kdim = config.n_embd
        self.key = nn.Linear(in_dim, config.n_embd)
        self.query = nn.Linear(in_dim, config.n_embd)
        self.value = nn.Linear(in_dim, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if self.causal:
            mask = torch.tril(torch.ones(config.block_size,
                                         config.block_size))
            if hasattr(config, "n_unmasked"):
                mask[:config.n_unmasked, :config.n_unmasked] = 1
            self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None, valid_mask=None):
        B, T, C = x.size()

        C = self.kdim
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        present = torch.stack((k, v))
        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat((past_key, k), dim=-2)
            v = torch.cat((past_value, v), dim=-2)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if layer_past is None and self.causal:
            att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))

        if valid_mask is not None:
            valid_mask = rearrange(valid_mask, 'b j -> b () () j')
            att = att.masked_fill_(~valid_mask, float('-inf'))

        att = nn.functional.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, present


class Base_STAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        assert dim == self.n_heads * self.head_dim, "dim must be divisible by n_heads"

        # 时间注意力参数
        self.temporal_key = nn.Linear(dim, dim)
        self.temporal_query = nn.Linear(dim, dim)
        self.temporal_value = nn.Linear(dim, dim)
        self.temporal_proj = nn.Linear(dim, dim)
        self.temporal_dropout = nn.Dropout(0.1)

        # 空间注意力参数
        self.spatial_key = nn.Linear(dim, dim)
        self.spatial_query = nn.Linear(dim, dim)
        self.spatial_value = nn.Linear(dim, dim)
        self.spatial_proj = nn.Linear(dim, dim)
        self.spatial_dropout = nn.Dropout(0.1)

        # 初始化因果掩码
        self.register_buffer("causal_mask", None)

    def _generate_causal_mask(self, T, device):
        """生成时间维度的因果掩码"""
        return torch.triu(
            torch.ones(T, T, device=device, dtype=torch.bool),
            diagonal=1
        )

    def forward(self, x):
        B, T, V, D = x.shape
        device = x.device

        # ==== 时间维度注意力 ====
        # 重塑输入为 [B*V, T, D]
        x_temp = x.permute(0, 2, 1, 3).reshape(B * V, T, D)

        # 计算Q, K, V
        k_t = self.temporal_key(x_temp).view(B * V, T, self.n_heads, self.head_dim).transpose(1, 2)  # [B*V, nh, T, hd]
        q_t = self.temporal_query(x_temp).view(B * V, T, self.n_heads, self.head_dim).transpose(1, 2)
        v_t = self.temporal_value(x_temp).view(B * V, T, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        att = (q_t @ k_t.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B*V, nh, T, T]

        # 应用因果掩码
        if self.causal_mask is None or self.causal_mask.size(0) < T:
            self.causal_mask = self._generate_causal_mask(T, device)
        att = att.masked_fill(self.causal_mask[:T, :T].unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax和投影
        att = torch.softmax(att, dim=-1)
        att = self.temporal_dropout(att)
        y_t = (att @ v_t).transpose(1, 2).reshape(B * V, T, D)  # [B*V, T, D]
        y_t = self.temporal_proj(y_t).view(B, V, T, D).permute(0, 2, 1, 3)  # [B, T, V, D]

        # ==== 空间维度注意力 ====
        # 重塑输入为 [B*T, V, D]
        x_spat = x.reshape(B * T, V, D)

        # 计算Q, K, V
        k_s = self.spatial_key(x_spat).view(B * T, V, self.n_heads, self.head_dim).transpose(1, 2)  # [B*T, nh, V, hd]
        q_s = self.spatial_query(x_spat).view(B * T, V, self.n_heads, self.head_dim).transpose(1, 2)
        v_s = self.spatial_value(x_spat).view(B * T, V, self.n_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        att = (q_s @ k_s.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))  # [B*T, nh, V, V]

        # Softmax和投影
        att = torch.softmax(att, dim=-1)
        att = self.spatial_dropout(att)
        y_s = (att @ v_s).transpose(1, 2).reshape(B * T, V, D)  # [B*T, V, D]
        y_s = self.spatial_proj(y_s).view(B, T, V, D)  # [B, T, V, D]

        return y_t + y_s


if __name__== "__main__":
    import torch
    from thop import profile
    from ptflops import get_model_complexity_info
    # 测试配置
    B = 1  # Batch size
    T = 64  # 时间步（帧数）
    V = 32  # 空间维度（关节点数）
    D = 64  # 特征维度
    n_heads = 8

    # 构造相同输入（三维和四维版本）
    x_3d = torch.randn(B, T, D)  # 用于 CausalSelfAttention
    x_4d = torch.randn(B, T, V, 16)  # 用于 STAttention


    # 创建模型实例
    class Config:
        n_embd = D
        n_head = n_heads
        causal = True
        block_size = T
        attn_pdrop = 0.1
        resid_pdrop = 0.1


    config = Config()

    # 实例化两个模型
    causal_attn = CausalSelfAttention(config)
    st_attn = Base_STAttention(16, 4)


    # 参数量计算
    def count_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    print(f"CausalSelfAttention 参数量: {count_params(causal_attn):,}")
    print(f"STAttention 参数量: {count_params(st_attn):,}")


    # 计算量测试（使用thop）
    def test_flops(model, input, name):
        flops, params = profile(model, inputs=(input,))
        print(f"{name} FLOPs: {flops:,}")


    # 测试 CausalSelfAttention（三维输入）
    test_flops(causal_attn, x_3d, "CausalSelfAttention")

    # 测试 STAttention（四维输入）
    test_flops(st_attn, x_4d, "STAttention")


    # 等价性验证（将四维输入转换为三维进行比较）
    def expand_input(x_4d):
        """将 [B,T,V,D] 转换为 [B*V, T, D]"""
        B, T, V, D = x_4d.shape
        return x_4d.permute(0, 2, 1, 3).reshape(B * V, T, D)



