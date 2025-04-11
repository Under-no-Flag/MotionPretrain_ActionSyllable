import torch
import torch.nn as nn


class ReConsLoss(nn.Module):
    def __init__(self, recons_loss):
        super(ReConsLoss, self).__init__()

        if recons_loss == 'l1':
            self.Loss = torch.nn.L1Loss()
        elif recons_loss == 'l2':
            self.Loss = torch.nn.MSELoss()
        elif recons_loss == 'l1_smooth':
            self.Loss = torch.nn.SmoothL1Loss()



    def forward(self, motion_pred, motion_gt):
        '''

        :param motion_pred: [B, T, V, 6]
        :param motion_gt: [B, T, V, 6]
        :return:
        '''
        # loss = self.Loss(motion_pred[..., : self.motion_dim], motion_gt[..., :self.motion_dim])
        loss = torch.mean(torch.norm(motion_pred - motion_gt, dim=-1, p=1))
        return loss


class GeometricConstraintLoss(nn.Module):
    """旋转矩阵几何约束损失模块(正交性+单位化)"""
    def __init__(self, lambda_ortho=5.0, lambda_unit=5.0):
        """
        :param lambda_ortho: 正交性约束权重
        :param lambda_unit: 单位化约束权重
        """
        super().__init__()
        self.lambda_ortho = lambda_ortho
        self.lambda_unit = lambda_unit

    def orthogonal_loss(self, vectors):
        """
        计算两组向量的正交性损失
        :param vectors: 输入张量[..., 6]，最后维前3为v1，后3为v2
        :return: 正交损失标量
        """
        v1, v2 = vectors[..., :3], vectors[..., 3:6]
        dot_products = torch.sum(v1 * v2, dim=-1)  # [..., ]
        return torch.mean(dot_products** 2)  # 平方均值损失

    def unit_loss(self, vectors):
        """
        计算向量单位化损失
        :param vectors: 输入张量[..., 6]
        :return: 单位化损失标量
        """
        v1_norm = torch.norm(vectors[..., :3], p=2, dim=-1)
        v2_norm = torch.norm(vectors[..., 3:6], p=2, dim=-1)
        return torch.mean((v1_norm - 1.0) ** 2 + (v2_norm - 1.0) ** 2)

    def forward(self, pred_vectors, gt_vectors=None):
        """
        计算总约束损失(默认仅作用于预测值)
        :param pred_vectors: 预测的向量[..., 6]
        :param gt_vectors: 真实向量(可选)，此处保留接口兼容性
        """
        loss_ortho = self.orthogonal_loss(pred_vectors)
        loss_unit = self.unit_loss(pred_vectors)
        return self.lambda_ortho * loss_ortho + self.lambda_unit * loss_unit

    def extra_repr(self):
        """打印配置参数"""
        return f"lambda_ortho={self.lambda_ortho}, lambda_unit={self.lambda_unit}"



