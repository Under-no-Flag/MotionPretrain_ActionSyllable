from torch import nn
import torch
import logging
from einops import rearrange
import math
from model.blocks.convolutions import Masked_conv,Masked_up_conv
from argparse import ArgumentParser
from model.quantize import VectorQuantizer,MCVectorQuantizer,ResidualVectorQuantizer,ResidualVectorQuantizerEMA,ResidualVectorQuantizerGCN,ResidualVectorQuantizerGCNEMA
import torch.nn.functional as F
import roma
from model.blocks.mingpt import STBlock
logger = logging.getLogger(__name__)
from model.transformer_vqvae import TransformerAutoEncoder
from model.blocks.down_up_sample import SpatioTemporalDown, SpatioTemporalUp,GA_Down,GA_Up

class MotionVQVAE(TransformerAutoEncoder):
    """
    针对运动数据的VQ-VAE，使用拓扑感知的MCVectorQuantizer
    """

    def __init__(self, *,
                 in_dim=6,
                 num_joints=32,
                 # 模型结构参数
                 n_layers=[0, 6], hid_dim=384, heads=4, dropout=0.1, causal_encoder=False, causal_decoder=False,
                 # 量化器特有参数
                 n_codebook=8,
                 n_e=1024, e_dim=128, beta=0.25,
                 mlp_hidden=256,  # 条件MLP的隐藏层
                 dataset_name='h36m',
                 ** kwargs):
        super().__init__(**{'n_layers': n_layers, 'hid_dim': hid_dim, 'e_dim': e_dim,
                            'in_dim': in_dim, 'heads': heads, 'dropout': dropout, 'causal_encoder': causal_encoder,
                            'causal_decoder': causal_decoder}, ** kwargs)

        # 验证维度兼容性
        self.seq_len = kwargs['seq_len']

        # 初始化可训练参数
        targets = ['body', 'root', 'trans', 'vert', 'fast_vert']
        self.log_sigmas = torch.nn.ParameterDict({k: torch.nn.Parameter(torch.zeros(1)) for k in targets})

        # 使用MCVectorQuantizer替换原量化器
        # self.quantizer = MCVectorQuantizer(
        #     num_joints=num_joints,
        #     n_e=n_e,
        #     e_dim=e_dim,
        #     beta=beta,
        #     nbooks=n_codebook,  # 对应关节数量
        #     mlp_hidden=mlp_hidden
        # )

        # self.quantizer = ResidualVectorQuantizerEMA(
        #     n_e=n_e,
        #     e_dim=e_dim,
        #     beta=0.25,
        #     n_q=n_codebook,  # 残差级数
        # )
        # self.quantizer = ResidualVectorQuantizer(
        #     n_e=n_e,
        #     e_dim=e_dim,
        #     beta=0.25,
        #     n_q=n_codebook,  # 残差级数
        # )
        from utils.data_utils import get_adjacency_matrix
        self.quantizer= ResidualVectorQuantizerGCNEMA(
            n_e=n_e,
            e_dim=e_dim,
            beta=0.25,
            n_q=n_codebook,  # 残差级数
            adjacency=get_adjacency_matrix(dataset_name)  # 使用H36M的邻接矩阵

        )



        self.n_e=n_e
        self.e_dim=e_dim
        self.n_codebook=n_codebook

    def forward_encoder(self, x, mask):
        """编码器前向传播，保持关节拓扑结构"""
        batch_size, seq_len, V, C = x.size()
        x = self.emb(x)  # 调整嵌入层输入
        hid, mask = self.encoder(x=x, mask=mask, return_mask=True)
        return hid.view(batch_size, seq_len, V, -1), mask  # 保持关节维度

    def forward_decoder(self, z, mask, return_mask=False):
        """解码器前向传播，处理拓扑量化结果"""
        batch_size, seq_len, V, C = z.size()
        return self.decoder(z=z, mask=mask, return_mask=return_mask)

    def forward(self, *, x, valid, quant_prop=1.0, ** kwargs):
        """完整前向传播，适配拓扑量化"""
        mask = valid
        batch_size, seq_len, V, C = x.size()

        # 编码阶段
        x = self.prepare_batch(x, mask)
        hid, mask_ = self.forward_encoder(x=x, mask=mask)


        # 量化处理
        z = hid
        # 掩码插值处理
        if mask_ is None:
            mask_ = F.interpolate(mask.float().reshape(batch_size, 1, seq_len, 1),
                                  size=[z.size(1), 1], mode='bilinear', align_corners=True)
            mask_ = mask_.long().reshape(batch_size, -1)
        else:
            mask_ = mask_.float().long()

        # 拓扑感知量化
        z_q, z_loss, indices = self.quantize(z, mask_, p=quant_prop)

        # 解码阶段
        hid = z_q  # [B, T, V, C]


        y = self.forward_decoder(z=hid, mask=mask_)

        # 输出处理
        rotmat = self.regressor(y)

        # 损失计算
        kl = math.log(self.n_e // self.n_codebook) * torch.ones_like(indices, requires_grad=False)

        return (rotmat,), {'quant_loss': z_loss, 'kl': kl, 'kl_valid': mask_}, indices
        # return (rotmat,), {'quant_loss': torch.Tensor([0.0]).cuda(), 'kl': 0, 'kl_valid': mask_}, 0

    def quantize(self, z, valid=None, p=1.0):
        """量化过程适配拓扑结构"""
        # 输入z形状: [B, T, V, C]
        z_q, loss, indices = self.quantizer(z, p=p)

        # 掩码处理
        if valid is not None:
            # valid_4d = valid.unsqueeze(-1)  # [B, T] -> [B, T, 1, 1]
            valid_4d = valid.unsqueeze(-1).unsqueeze(-1)  # [B, T] -> [B, T, 1, 1]
            loss = (loss * valid_4d).sum() / valid_4d.sum()
            indices = indices * valid_4d.long() - (1 - valid_4d.long())
        return z_q, loss, indices

    def forward_from_indices(self, indices: torch.Tensor):
        """
        Decode RVQ codebook indices back to motion.
        indices : [B,T,V]      – for single-level VQ
                  [B,T,V,n_q]  – for multi-level RVQ / Pyramid-RVQ
        """
        # ---- 1. 兼容单级 VQ -----------------------------------
        if indices.dim() == 3:
            indices = indices.unsqueeze(-1)  # -> [B,T,V,1]

        # ---- 2. 获取量化向量 ---------------------------------
        z_q = self.quantizer.get_codebook_entry(indices)  # [B,T,V,e_dim]

        # ---- 3. 解码 ----------------------------------------
        B, T = indices.shape[0], indices.shape[1]
        mask = torch.ones(B, T, dtype=torch.long, device=indices.device)
        y = self.forward_decoder(z=z_q, mask=mask)

        # ---- 4. 回到旋转矩阵 ---------------------------------
        rotmat = self.regressor(y)
        return rotmat

    @staticmethod
    def add_model_specific_args(parent_parser):
        """添加拓扑量化特有参数"""
        parser = super(MotionVQVAE, MotionVQVAE).add_model_specific_args(parent_parser)
        # 量化器参数
        parser.add_argument("--n_codebook", type=int, default=11,
                            help="关节数量，需与运动链定义匹配")
        parser.add_argument("--n_e", type=int, default=1024)
        parser.add_argument("--e_dim", type=int, default=128)
        parser.add_argument("--beta", type=float, default=0.25)
        # 拓扑结构参数
        parser.add_argument("--motion_chain", type=eval, default="[[0,1,2,3,4,5],[6,7,8,9,10]]",
                            help="运动链定义，使用嵌套列表格式")
        parser.add_argument("--mlp_hidden", type=int, default=256,
                            help="条件MLP隐藏层维度")
        return parser


class PyramidalTAE(TransformerAutoEncoder):
    """
    针对运动数据的VQ-VAE，使用拓扑感知的MCVectorQuantizer
    """

    def __init__(self, *, in_dim=1024,
                 num_joints=32,
                 n_layers=[0, 6],
                 hid_dim=384,
                 heads=4, dropout=0.1,
                 causal_encoder=False,
                 causal_decoder=False,
                 n_codebook=8,
                 n_e=1024,
                 e_dim=128,
                 beta=0.25,
                 **kwargs):
        super().__init__(**{'n_layers': n_layers, 'hid_dim': hid_dim, 'e_dim': e_dim,
                            'in_dim': in_dim, 'heads': heads, 'dropout': dropout, 'causal_encoder': causal_encoder,
                            'causal_decoder': causal_decoder}, **kwargs)

        # 验证维度兼容性
        self.seq_len = kwargs['seq_len']

        # self.downsample_layer = SpatioTemporalDown(
        #     c_in=hid_dim,
        #     c_out=hid_dim,
        # )
        # self.upsample_layer = SpatioTemporalUp(
        #     c_in=hid_dim,
        #     c_out=hid_dim,
        # )


        self.downsample_layer = GA_Down(
            V=num_joints,
            P=num_joints//2,
            C=hid_dim,
        )
        self.upsample_layer = GA_Up(
            V=num_joints,
            P=num_joints//2,
            C=hid_dim,
        )

        self.quantizer = ResidualVectorQuantizer(
            n_e=n_e,
            e_dim=e_dim,
            beta=0.25,
            n_q=n_codebook,  # 残差级数
        )

        self.n_e = n_e
        self.e_dim = e_dim
        self.n_codebook = n_codebook

    def forward_encoder(self, x, mask):
        """编码器前向传播，保持关节拓扑结构"""
        x = self.emb(x)  # 调整嵌入层输入
        x,attn=self.downsample_layer(x)
        batch_size, seq_len, V, C = x.size()
        hid, mask = self.encoder(x=x, mask=mask, return_mask=True)
        return hid.view(batch_size, seq_len, V, -1), mask,attn  # 保持关节维度

    def forward_decoder(self, z, mask, return_mask=False):
        """解码器前向传播，处理拓扑量化结果"""
        batch_size, seq_len, V, C = z.size()
        return self.decoder(z=z, mask=mask, return_mask=return_mask)

    def forward(self, *, x, valid, quant_prop=1.0, **kwargs):
        """完整前向传播，适配拓扑量化"""
        mask = valid
        _, T_0, V_0,C = x.size()
        # 编码阶段
        x = self.prepare_batch(x, mask)
        hid, mask_ ,attn= self.forward_encoder(x=x, mask=mask)

        # 量化处理
        z=hid
        batch_size, seq_len, *_ = z.size()

        # 掩码插值处理



        # 拓扑感知量化
        z_q, z_loss, indices = self.quantize(z, mask_[...,:seq_len], p=quant_prop)

        # 解码阶段

        y = self.forward_decoder(z=z_q, mask=mask_)
        # y = self.upsample_layer(y,T_0,V_0)  # [B, T, V, C]
        y = self.upsample_layer(y,attn,T_0)  # [B, T, V, C]


        # 输出处理
        rotmat = self.regressor(y)

        # 损失计算
        kl = math.log(self.n_e // self.n_codebook) * torch.ones_like(indices, requires_grad=False)

        return (rotmat,), {'quant_loss': z_loss, 'kl': kl, 'kl_valid': mask_}, indices

    def quantize(self, z, valid=None, p=1.0):
        """量化过程适配拓扑结构"""
        # 输入z形状: [B, T, V, C]
        z_q, loss, indices = self.quantizer(z, p=p)

        # 掩码处理
        if valid is not None:
            valid_4d = valid.unsqueeze(-1).unsqueeze(-1)  # [B, T] -> [B, T, 1, 1]
            loss = (loss * valid_4d).sum() / valid_4d.sum()
            indices = indices * valid_4d.long() - (1 - valid_4d.long())
        return z_q, loss, indices

    def forward_from_indices(self, indices: torch.Tensor):
        """
        Decode RVQ codebook indices back to motion.
        indices : [B,T,V]      – for single-level VQ
                  [B,T,V,n_q]  – for multi-level RVQ / Pyramid-RVQ
        """
        # ---- 1. 兼容单级 VQ -----------------------------------
        if indices.dim() == 3:
            indices = indices.unsqueeze(-1)  # -> [B,T,V,1]

        # ---- 2. 获取量化向量 ---------------------------------
        z_q = self.quantizer.get_codebook_entry(indices)  # [B,T,V,e_dim]

        # ---- 3. 解码 ----------------------------------------
        B, T = indices.shape[0], indices.shape[1]
        mask = torch.ones(B, T, dtype=torch.long, device=indices.device)
        y = self.forward_decoder(z=z_q, mask=mask)

        # ---- 4. 回到旋转矩阵 ---------------------------------
        rotmat = self.regressor(y)
        return rotmat

    @staticmethod
    def add_model_specific_args(parent_parser):
        """添加拓扑量化特有参数"""
        parser = super(MotionVQVAE, MotionVQVAE).add_model_specific_args(parent_parser)
        # 量化器参数
        parser.add_argument("--n_codebook", type=int, default=11,
                            help="关节数量，需与运动链定义匹配")
        parser.add_argument("--n_e", type=int, default=1024)
        parser.add_argument("--e_dim", type=int, default=128)
        parser.add_argument("--beta", type=float, default=0.25)
        # 拓扑结构参数
        parser.add_argument("--motion_chain", type=eval, default="[[0,1,2,3,4,5],[6,7,8,9,10]]",
                            help="运动链定义，使用嵌套列表格式")
        parser.add_argument("--mlp_hidden", type=int, default=256,
                            help="条件MLP隐藏层维度")
        return parser                                 # 回到 (B,T,V,C)

if __name__=="__main__":
    device=torch.device("cuda:0")
    model=MotionVQVAE(
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
        n_layers=[0,6],
        seq_len=64,
        dataset_name="3dpw"
        ).to(device)
    print(model)
    #计算模型参数量
    print("MotionVQVAE模型参数量:", sum(p.numel() for p in model.parameters())/1e6, "M")
    input=torch.randn(16,64,24,6).to(device)
    import time
    start=time.time()
    output=model(x=input,valid=torch.ones(16,64).to(device))
    print("MotionVQVAE量化耗时:", time.time()-start)
    print(output[0][0].shape)
    from thop import profile


    # 定义包装类
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x, y, valid):  # 接受位置参数
            # 转换为关键字参数调用原模型
            return self.model(x=x, y=y, valid=valid)


    wrapped_model = ModelWrapper(model)
    # 计算 FLOPs 和参数量
    macs, params = profile(wrapped_model, inputs=(input, 0, torch.ones(1,64).to(device)),verbose=False)
    print(f"Params: {params / 1e6} M, FLOPs: {macs / 1e6} M")

    model = PyramidalTAE(
        in_dim=6, hid_dim=128, n_layers=[0, 6],
        n_head=4,
        down_cfg=((2, 2),),  # 一层时空↓2
        num_joints=32, seq_len=64,
        n_codebook=8, n_e=256, e_dim=128).cuda()
    #计算模型参数量
    print("PyramidalTAE模型参数量:", sum(p.numel() for p in model.parameters())/1e6, "M")
    B, T, V, C = 16, 64, 32, 6
    x = torch.randn(B, T, V, C).cuda()

    import time
    start=time.time()
    out, losses, _ = model(x=x, valid=torch.ones(B, T).long().cuda())
    print("PyramidalTAE量化耗时:", time.time()-start)
    print(out[0].shape)  # (B,T,V,in_dim) – 完全与旧接口一致
    wrapped_model = ModelWrapper(model)
    # 计算 FLOPs 和参数量
    macs, params = profile(wrapped_model, inputs=(input, 0, torch.ones(1,64).to(device)),verbose=False)
    print(f"Params: {params / 1e6} M, FLOPs: {macs / 1e6} M")