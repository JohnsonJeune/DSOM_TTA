import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from . import memory
import torch.nn.functional as F
import numpy as np
import copy
# https://discuss.pytorch.org/t/calculating-the-entropy-loss/14510
# but there is a bug in the original code: it sums up the entropy over a batch. so I take mean instead of sum
class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        #print('这里使用了HLOSS')
        softmax = F.softmax(x/self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax+1e-6)
        b = entropy.mean()

        return b

class NOTE(nn.Module):
    """NOTE: Neural Online Test-time Adaptation with Entropy Minimization"""

    def __init__(self, args, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.args = args
        self.model = configure_model(model, args)
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)

        self.mem = memory.build_memory(args)
        self.fifo = memory.FIFO(capacity=args.update_every_x)
        self.entropy_loss = HLoss(temp_factor=args.temperature)

    def forward(self, x, current_num_sample=None):
        """
        x: Tensor, shape (B,1,T,C)
        current_num_sample: 当前批次第一个样本编号（可选）
        """
        if self.episodic:
            self.reset()

        B = x.size(0)

        # 如果 memory 还没初始化，用第一个样本初始化一次
        if self.mem.get_occupancy() == 0:
            with torch.no_grad():
                f = x[0,:,:,:]          # (1,T,C)

                self.update_memory(f, current_num_sample=1)
                #print("[NOTE] Initialized memory from batch x[0]")

            # 若 batch 大于1，后续样本从索引1开始
            start_idx = 1
        else:
            start_idx = 0

        # 其余样本逐个加入 memory
        for i in range(start_idx, B):
            f = x[i,:,:,:]            # (1,T,C)

            current_idx = current_num_sample + i if current_num_sample is not None else None
            self.update_memory(f, current_num_sample=current_idx)

        # batch-wise 只做一次 adaptation
        outputs, _ = forward_and_adapt_note(x, self.model, self.optimizer, self.mem, self.entropy_loss, self.args)

        return outputs




    def update_memory(self, batch_x, current_num_sample=None):
        """
        批量更新 NOTE memory buffers，batch_x 形状为 (B, 1, T, C)。

        Args:
            batch_x (Tensor): 输入特征，形状 (B, 1, T, C)
            current_num_sample (int, optional): 当前样本编号（可选，用于判断是否触发评估）
        """

        B = batch_x.size(0)

        # 遍历 batch 中每个样本
        for i in range(B):
            f = batch_x[i].squeeze(0)  # (T, C)

            # 先加入 fifo（如果有）
            self.fifo.add_instance([f, torch.tensor(0), torch.tensor(0)])  # c, d 默认占位 0

            with torch.no_grad():
                self.model.eval()

                if self.args.memory_type in ['FIFO', 'Reservoir']:
                    # 伪标签暂时用0，后续也可以在这里用模型推理获取
                    f_device = f.to(self.args.device)
                    logit, _ = self.model(f_device.unsqueeze(0).unsqueeze(1))
                    pseudo_cls = logit.argmax(dim=1)[0].cpu()
                    self.mem.add_instance([f, pseudo_cls, torch.tensor(0)])
                
                elif self.args.memory_type == 'PBRS':
                    f_device = f.to(self.args.device)
                    logit, _ = self.model(f_device.unsqueeze(0).unsqueeze(1))
                    pseudo_cls = logit.argmax(dim=1)[0].cpu()
                    d = torch.tensor(0)  # 域标签占位
                    c = torch.tensor(0)  # 真实标签占位
                    self.mem.add_instance([f, pseudo_cls, d, c, 0])

        # 评估策略
        if self.args.use_learned_stats:
            # 评估单个样本时要改成评估 batch，这里简单传入当前 batch
            self.evaluate([batch_x, torch.zeros(B), torch.zeros(B)])  # 标签和域标签都用 0 占位
        elif current_num_sample is not None and current_num_sample % self.args.update_every_x == 0:
            self.evaluate(self.fifo.get_memory())



    def evaluate(self, data):
        """
        Optional hook for online evaluation. Override or link to evaluation_online.
        """
        pass

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt_note(x, model, optimizer, memory, entropy_loss, args):
    """
    Batch-wise NOTE: Forward and adapt using entropy minimization.
    x: Tensor (B, 1, T, C)
    """
    model.train()
    x = x.squeeze(1)  # (B, T, C)

    # ⚠️ 若 memory 中样本不足，不执行适应
    if args.no_adapt or memory.get_occupancy() < args.update_every_x:
        with torch.no_grad():
            outputs, _ = model(x.unsqueeze(1))
            return outputs, 0

    # ✅ Memory 取全部样本组成训练大 batch
    feats, _, _ = memory.get_memory()
    feats = torch.stack(feats).to(x.device)  # (M, T, C)
    dataset = torch.utils.data.TensorDataset(feats)
    data_loader = DataLoader(dataset, batch_size=args.n_batch_size,
                                 shuffle=True, drop_last=False, pin_memory=False)
    for e in range(args.epoch):
            # 迭代数据加载器中的每个批次
            for batch_idx, (feats,) in enumerate(data_loader):
                # 将特征移动到指定设备
                feats = feats.to(x.device)
                # 前向传播更新 BN 层统计信息
                preds_of_data, _ = model(feats.unsqueeze(1))

                # 计算损失
                loss = entropy_loss(preds_of_data)
                # 梯度清零
                optimizer.zero_grad()
                # 反向传播计算梯度
                loss.backward()
                # 更新模型参数
                optimizer.step()


    # ✅ 最终返回当前输入 batch 的预测（inference only）
    with torch.no_grad():
        model.eval()
        output, _ = model(x.unsqueeze(1))
        return output, 0



def configure_model(model, args):
    """Configure model for NOTE adaptation."""
    model.train()
    model.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
            if args.use_learned_stats:
                module.track_running_stats = True
                module.momentum = args.bn_momentum
            else:
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)

        elif isinstance(module, nn.InstanceNorm1d) or isinstance(module, nn.InstanceNorm2d):
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)

        if args.iabn:
            if isinstance(module, InstanceAwareBatchNorm1d) or isinstance(module, InstanceAwareBatchNorm2d):
                for param in module.parameters():
                    param.requires_grad = True

    return model



def copy_model_and_optimizer(model, optimizer):
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)



def convert_iabn(module, args):
    """
    递归地将 nn.BatchNorm1d/2d 替换为 InstanceAwareBatchNorm。
    参数完全从 args 传入。
    """
    module_output = module
    if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
        IABN = InstanceAwareBatchNorm2d if isinstance(module, nn.BatchNorm2d) else InstanceAwareBatchNorm1d
        module_output = IABN(
            num_channels=module.num_features,
            k=args.iabn_k,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            skip_thres=args.skip_thres
        )
        module_output._bn = copy.deepcopy(module)

    for name, child in module.named_children():
        module_output.add_module(
            name, convert_iabn(child, args)
        )

    return module_output


class InstanceAwareBatchNorm2d(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True, skip_thres=0):
        super().__init__()
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.affine = affine
        self.skip_thres = skip_thres
        self._bn = nn.BatchNorm2d(num_channels, eps=eps, momentum=momentum, affine=affine)

    def _softshrink(self, x, lbd):
        return F.relu(x - lbd) - F.relu(-(x + lbd))

    def forward(self, x):
        #print('是否为BN2D')
        b, c, h, w = x.size()
        sigma2, mu = torch.var_mean(x, dim=[2, 3], keepdim=True, unbiased=True)

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
        else:
            if not self._bn.track_running_stats and self._bn.running_mean is None:
                sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2, 3], keepdim=True, unbiased=True)
            else:
                mu_b = self._bn.running_mean.view(1, c, 1, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1, 1)

        if h * w <= self.skip_thres:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / (h * w))
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (h * w - 1))
            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
            sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)
            sigma2_adj = F.relu(sigma2_adj)

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._bn.weight.view(c, 1, 1)
            bias = self._bn.bias.view(c, 1, 1)
            x_n = x_n * weight + bias

        return x_n


class InstanceAwareBatchNorm1d(nn.Module):
    def __init__(self, num_channels, k=3.0, eps=1e-5, momentum=0.1, affine=True, skip_thres=0):
        super().__init__()
        self.num_channels = num_channels
        self.k = k
        self.eps = eps
        self.affine = affine
        self.skip_thres = skip_thres
        self._bn = nn.BatchNorm1d(num_channels, eps=eps, momentum=momentum, affine=affine)

    def _softshrink(self, x, lbd):
        return F.relu(x - lbd) - F.relu(-(x + lbd))

    def forward(self, x):
        #print('是否为BN1D')
        b, c, l = x.size()
        sigma2, mu = torch.var_mean(x, dim=2, keepdim=True, unbiased=True)

        if self.training:
            _ = self._bn(x)
            sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
        else:
            if not self._bn.track_running_stats and self._bn.running_mean is None:
                sigma2_b, mu_b = torch.var_mean(x, dim=[0, 2], keepdim=True, unbiased=True)
            else:
                mu_b = self._bn.running_mean.view(1, c, 1)
                sigma2_b = self._bn.running_var.view(1, c, 1)

        if l <= self.skip_thres:
            mu_adj = mu_b
            sigma2_adj = sigma2_b
        else:
            s_mu = torch.sqrt((sigma2_b + self.eps) / l)
            s_sigma2 = (sigma2_b + self.eps) * np.sqrt(2 / (l - 1))
            mu_adj = mu_b + self._softshrink(mu - mu_b, self.k * s_mu)
            sigma2_adj = sigma2_b + self._softshrink(sigma2 - sigma2_b, self.k * s_sigma2)
            sigma2_adj = F.relu(sigma2_adj)

        x_n = (x - mu_adj) * torch.rsqrt(sigma2_adj + self.eps)

        if self.affine:
            weight = self._bn.weight.view(c, 1)
            bias = self._bn.bias.view(c, 1)
            x_n = x_n * weight + bias

        return x_n


def collect_params(model):
    """
    收集所有 BatchNorm 和 InstanceAwareBatchNorm 中的 weight 和 bias。
    返回两个列表：参数对象 和 参数名（可选用于调试）。
    """
    params = []
    names = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d,
                               InstanceAwareBatchNorm1d, InstanceAwareBatchNorm2d)):
            for pname, p in module.named_parameters():
                if pname in ['weight', 'bias']:
                    if p.requires_grad:
                        params.append(p)
                        names.append(f"{name}.{pname}")
    return params, names
