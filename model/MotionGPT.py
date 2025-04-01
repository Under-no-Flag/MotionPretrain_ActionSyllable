import torch
from torch import nn


class SpatioTemporalDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.st_attn = SpatioTemporalAttention(d_model, n_heads)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # 时空注意力
        attn_out = self.st_attn(x)
        x = self.norm1(x + attn_out)

        # 前馈网络
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class Decoder(nn.Module):
    def __init__(self, d_model, n_heads, num_layers, max_seq_len):
        super().__init__()
        self.pos_embed = nn.Embedding(max_seq_len, d_model)
        self.layers = nn.ModuleList([
            SpatioTemporalDecoderLayer(d_model, n_heads)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        # x: [B, T, V, C]
        B, T, V, C = x.shape

        # 生成位置编码
        positions = torch.arange(T, device=x.device).expand(B, V, T).permute(0, 2, 1)
        pos_emb = self.pos_embed(positions)  # [B, T, V, C]

        # 添加位置编码
        x = x + pos_emb

        # 通过解码层
        for layer in self.layers:
            x = layer(x)
        return x


class SpatioTemporalAttention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(dim, n_heads)
        self.spatial_attn = nn.MultiheadAttention(dim, n_heads)
        self.merge = nn.Linear(2 * dim, dim)

        # 初始化因果掩码（动态生成）
        self.causal_mask = None

    def _generate_causal_mask(self, seq_len, device):
        """生成时间维度的因果掩码"""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1
        )

    def forward(self, x):
        # x: [Batch, Frames, Joints, Dim]
        B, T, V, D = x.shape
        device = x.device

        # ==== 时间维度注意力 ====
        # 生成因果掩码
        if self.causal_mask is None or self.causal_mask.size(0) != T:
            self.causal_mask = self._generate_causal_mask(T, device)

        # 重塑输入并应用时间注意力
        temp = x.permute(1, 0, 2, 3).reshape(T, B * V, D)
        temp, _ = self.temporal_attn(
            query=temp,
            key=temp,
            value=temp,
            attn_mask=self.causal_mask
        )
        temp = temp.view(T, B, V, D).permute(1, 0, 2, 3)

        # ==== 空间维度注意力 ====
        spat = x.reshape(B * T, V, D)
        spat, _ = self.spatial_attn(spat, spat, spat)
        spat = spat.view(B, T, V, D)

        # 融合时空特征
        combined = torch.cat([temp, spat], dim=-1)
        return self.merge(combined)

class MotionGPT(nn.Module):
    def __init__(self, d_model=64, n_heads=4, num_layers=6,
                 max_seq_len=64, input_dim=6, output_dim=3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.decoder = Decoder(d_model, n_heads, num_layers, max_seq_len)
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        # x: [B, T, V, C] 输入骨架序列
        x = self.input_proj(x)  # 投影到模型维度

        # 通过时空解码器
        x = self.decoder(x)  # [B, T, V, D]

        # 预测下一帧的3D坐标
        return self.output_proj(x)  # [B, T, V, 6]


class MotionLoss(nn.Module):
    def __init__(self):
        super(MotionLoss,self).__init__()

    def forward(self, pred, gt):
        # 计算多关节均方角度误差
        return torch.mean(torch.norm(pred - gt, dim=-1, p=2))




if __name__ == "__main__":
    # 测试样例
    B, T, V, C = 1, 64, 50, 6  # Batch, 帧数, 关节数, 输入维度(6D旋转)
    model = MotionGPT(input_dim=6, output_dim=6,num_layers=10,d_model=64)
    input_seq = torch.randn(B, T, V, C)

    # 预测下一帧的3D坐标
    output = model(input_seq)
    print("预测坐标形状:", output.shape)  # 应为 [16, 10, 24, 6]

    loss=MotionLoss()(output,input_seq)
    print("Loss:",loss.shape)
    from thop import profile, clever_format
    flops, params = profile(model, inputs=(input_seq,))

    print('FLOPs:', flops / 1e6, 'M')  # 将FLOPs转换为G（十亿）
    print('Params:', params / 1e6, 'M')  # 将参数量转换为M（百万）