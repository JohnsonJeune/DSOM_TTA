"""
Builds upon: https://github.com/fiveai/LAME
Corresponding paper: https://arxiv.org/abs/2201.05718
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


"""
Builds upon: https://github.com/fiveai/LAME
Corresponding paper: https://arxiv.org/abs/2201.05718
"""
def get_classifier(args, model):

    return model.classifier

class LAME(nn.Module):
    def __init__(self, args, model, optimizer=None):
        super().__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer  # LAME 不使用 optimizer，但为了统一格式保留

        self.force_symmetry = args.FORCE_SYMMETRY
        self.affinity = eval(f'{args.AFFINITY}_affinity')(
            sigma=args.SIGMA,
            knn=args.KNN
        )


        # 拆分模型结构
        self.feature_extractor = self.model
        self.classifier = get_classifier(self.args, self.model)

        # 拷贝模型状态用于 reset
        self.model_state = deepcopy(self.model.state_dict())

        self.configure_model()

    def forward(self, x):
        return self.forward_and_adapt(x)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        """
        前向传播并执行基于标签传播的自适应过程
        利用特征 affinity 矩阵和拉普拉斯优化实现测试时自适应
        
        Args:
            x (torch.Tensor): 输入测试图像张量，形状通常为 [B, C, H, W]
            
        Returns:
            torch.Tensor: 经过标签传播优化后的分类概率分布，形状为 [B, num_classes]
        """
        # 将输入重命名为更具描述性的变量，明确表示为测试图像
        imgs_test = x

        # 从测试图像中提取特征，忽略特征提取器的第一个返回值
        # 假设 feature_extractor 返回 (logits, features) 或类似格式
        _ ,features = self.feature_extractor(imgs_test)
        
        # 使用分类器对提取的特征进行初始分类，获取原始输出
        outputs = self.classifier(features)

        # 计算一元势能 (unary potential)，用于标签传播
        # 公式：-log(softmax概率 + 小常数)，添加1e-10避免数值不稳定
        unary = - torch.log(outputs.softmax(dim=1) + 1e-10)

        # 构建 affinity 核矩阵（相似度矩阵）
        # 对特征进行L2归一化，确保相似度计算不受特征尺度影响
        features = F.normalize(features, p=2, dim=-1)
        # 基于归一化特征计算affinity矩阵（相似度矩阵）
        kernel = self.affinity(features)
        # 如果需要强制对称性，通过核矩阵与其转置的平均来实现
        if self.force_symmetry:
            kernel = 0.5 * (kernel + kernel.t())

        # 标签传播优化：通过拉普拉斯优化算法更新输出
        # 结合一元势能和affinity核矩阵进行标签传播，得到优化后的分类结果
        outputs = laplacian_optimization(unary, kernel)
        
        # 返回经过自适应调整后的输出
        return outputs

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):
    """
    基于拉普拉斯能量最小化的标签传播优化算法
    通过迭代更新分类概率分布，最小化包含一元势能、成对势能和熵项的能量函数
    
    Args:
        unary (torch.Tensor): 一元势能矩阵，形状为 [N, K]，N为样本数，K为类别数
        kernel (torch.Tensor): Affinity相似度矩阵，形状为 [N, N]，表示样本间的相似度
        bound_lambda (float): 成对势能的权重系数，控制成对项对能量函数的影响程度
        max_steps (int): 最大迭代次数，防止优化过程发散
        
    Returns:
        torch.Tensor: 优化后的分类概率分布，形状为 [N, K]
    """
    # 存储每次迭代的能量值，用于监控收敛过程
    E_list = []
    # 初始化上一轮能量值为无穷大，用于收敛判断
    oldE = float('inf')
    # 初始化概率分布Y：从一元势能的负对数转换为初始概率（softmax归一化）
    Y = (-unary).softmax(-1)  # [N, K]
    
    # 迭代优化过程
    for i in range(max_steps):
        # 计算成对势能：bound_lambda * 相似度矩阵 * 当前概率分布
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        # 指数项：-一元势能 + 成对势能（综合两类势能影响）
        exponent = -unary + pairwise
        # 更新概率分布Y：对指数项进行softmax归一化
        Y = exponent.softmax(-1)
        # 计算当前能量值并存储（调用entropy_energy计算综合能量）
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        # 收敛判断：当迭代次数>1且能量变化小于阈值（相对变化<1e-8）时停止迭代
        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            # logger.info(f'Converged in {i} iterations')  # 收敛日志（当前注释掉）
            break
        else:
            # 更新上一轮能量值，准备下一次迭代
            oldE = E

    # 返回优化后的概率分布
    return Y

def entropy_energy(Y, unary, pairwise, bound_lambda):
    """
    计算拉普拉斯能量函数，综合考虑一元势能、成对势能和熵正则化项
    
    能量函数公式：E = sum( unary*Y - bound_lambda*pairwise*Y + Y*log(Y) )
    其中：
    - unary*Y：一元势能项，惩罚与初始分类不一致的分布
    - bound_lambda*pairwise*Y：成对势能项，鼓励相似样本有相似分布
    - Y*log(Y)：熵正则化项，鼓励分布的平滑性
    
    Args:
        Y (torch.Tensor): 当前分类概率分布，形状 [N, K]
        unary (torch.Tensor): 一元势能矩阵，形状 [N, K]
        pairwise (torch.Tensor): 成对势能矩阵，形状 [N, K]
        bound_lambda (float): 成对势能的权重系数
        
    Returns:
        torch.Tensor: 标量能量值
    """
    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clip(1e-20))).sum()
    return E

class AffinityMatrix:

    def __init__(self, **kwargs):
        pass

    def __call__(self, X, **kwargs):
        raise NotImplementedError

    def is_psd(self, mat):
        eigenvalues = torch.eig(mat)[0][:, 0].sort(descending=True)[0]
        return eigenvalues, float((mat == mat.t()).all() and (eigenvalues >= 0).all())

    def symmetrize(self, mat):
        return 1 / 2 * (mat + mat.t())

class kNN_affinity(AffinityMatrix):
    def __init__(self, knn: int, **kwargs):
        self.knn = knn

    def __call__(self, X):
        N = X.size(0)
        dist = torch.cdist(X, X, p=2)  # [N, N]
        n_neighbors = min(self.knn + 1, N)

        knn_index = dist.topk(n_neighbors, -1, largest=False).indices[:, 1:]  # [N, knn]

        W = torch.zeros(N, N, device=X.device)
        W.scatter_(dim=-1, index=knn_index, value=1.0)

        return W


class rbf_affinity(AffinityMatrix):
    def __init__(self, sigma: float, **kwargs):
        self.sigma = sigma
        self.k = kwargs['knn']

    def __call__(self, X):

        N = X.size(0)
        dist = torch.cdist(X, X, p=2)  # [N, N]
        n_neighbors = min(self.k, N)
        kth_dist = dist.topk(k=n_neighbors, dim=-1, largest=False).values[:, -1]  # compute k^th distance for each point, [N, knn + 1]
        sigma = kth_dist.mean()
        rbf = torch.exp(- dist ** 2 / (2 * sigma ** 2))
        # mask = torch.eye(X.size(0)).to(X.device)
        # rbf = rbf * (1 - mask)
        return rbf

class linear_affinity(AffinityMatrix):

    def __call__(self, X: torch.Tensor):
        """
        X: [N, d]
        """
        return torch.matmul(X, X.t())
