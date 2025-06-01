
from torch import nn
import torch
from einops import rearrange


# Part-aware Motion Tokeniser (PaMT)
class PaMT(nn.Module):
    def __init__(self,
                 num_parts=10,
                 hidden=128,
                 C_in=6,
                 k_t=3,
                 s_t=2,
                 ):
        super().__init__()
        self.num_parts = num_parts
        # ②   可学习“关节→部件”映射   S ∈ R^{V×P}
        self.attn_score = nn.Linear(hidden, num_parts, bias=False)
        # ③   Temporal 1-D depthwise-conv 做时域下采样
        self.temp_conv = nn.Conv1d(hidden, hidden,
                                   kernel_size=k_t, stride=s_t, groups=hidden)
    def forward(self, x):          # x:(B,T,V,C)
        α = torch.softmax(self.attn_score(x), dim=2)        # (B,T,V,P)
        part_tok = (α.transpose(2,3) @ x) / (α.sum(2, keepdim=True)+1e-6)
        #       (B,T,P,H)
        part_tok = rearrange(part_tok, 'b t p h -> (b p) h t')
        part_tok = self.temp_conv(part_tok)                 # 时域↓
        part_tok = rearrange(part_tok, '(b p) h t -> b t p h', p=self.num_parts)
        return part_tok          # (B,Tʹ,P,H)

if __name__ == '__main__':
    # 测试
    B, T, V, C = 2, 128, 25, 3

