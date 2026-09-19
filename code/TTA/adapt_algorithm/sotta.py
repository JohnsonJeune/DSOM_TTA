import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from copy import deepcopy
from .memory import *

from .optimizer import SAM, sam_collect_params
import torch.optim as optim




class SoTTA(nn.Module):
    def __init__(self, args, model, optimizer, steps=1, episodic=False):
        super().__init__()
        self.args = args
        self.model = configure_model(model, args)
        model_params, _ = sam_collect_params(model)

        # 实例化 SAM 优化器
        self.optimizer = SAM(
        model_params, 
        base_optimizer=optim.SGD, 
        rho=0.05, 
        adaptive=True,  # 直接写死为 True 或 False
        lr=0.001,
        momentum=0.9
    )

        self.steps = steps
        self.episodic = episodic

        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        self.mem = build_memory(args)
        self.fifo = FIFO(capacity=args.update_every_x)
        self.entropy_loss = HLoss(temp_factor=args.temperature)

    def forward(self, x, current_num_sample=None):
        """
        x: Tensor (B, T, C)
        """
        if self.episodic:
            self.reset()

        B = x.size(0)
        if self.mem.get_occupancy() == 0:
            with torch.no_grad():
                self.update_memory(x[0].unsqueeze(0), current_num_sample=1)
            start_idx = 1
        else:
            start_idx = 0

        for i in range(start_idx, B):
            current_idx = current_num_sample + i if current_num_sample else None
            self.update_memory(x[i].unsqueeze(0), current_num_sample=current_idx)

        outputs, _ = forward_and_adapt_sotta(self, x)
        return outputs

    def update_memory(self, batch_x, current_num_sample=None):
        B = batch_x.size(0)
        for i in range(B):
            f = batch_x[i].squeeze(0)
            self.fifo.add_instance([f, torch.tensor(0), torch.tensor(0)])

            with torch.no_grad():
                self.model.eval()
                if self.args.memory_type == 'FIFO':
                    self.mem.add_instance([f, torch.tensor(0), torch.tensor(0)])
                elif self.args.memory_type in ['HUS', 'ConfFIFO']:
                    f_device = f.to(self.args.device)
                    logit,_ = self.model(f_device.unsqueeze(0).unsqueeze(1))
                    pseudo_cls = logit.argmax(1)[0].cpu()
                    pseudo_conf = F.softmax(logit, dim=1).max(1)[0][0].cpu()
                    self.mem.add_instance([f, pseudo_cls, torch.tensor(0), pseudo_conf])
                elif self.args.memory_type == 'CSTU':
                    f_device = f.to(self.args.device)
                    out,_ = self.model(f_device.unsqueeze(0))
                    prob = torch.softmax(out, dim=1)
                    pseudo_label = torch.argmax(prob, dim=1)
                    entropy = torch.sum(-prob * torch.log(prob + 1e-6), dim=1)
                    self.mem.add_instance([f, pseudo_label[0], entropy[0]])

        if self.args.use_learned_stats:
            self.evaluate([batch_x, torch.zeros(B), torch.zeros(B)])
        elif current_num_sample and current_num_sample % self.args.update_every_x == 0:
            self.evaluate(self.fifo.get_memory())

    def evaluate(self, data):
        """
        Optional evaluation hook.
        """
        pass

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("Cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt_sotta(self, x):
    """
    Batch-wise adaptation step for SoTTA using self.model, self.optimizer, self.mem, etc.
    """
    self.model.train()

    if self.args.no_adapt or self.mem.get_occupancy() < self.args.update_every_x:
        with torch.no_grad():
            output,_ = self.model(x)
            return output, 0

    # 从 memory 获取样本
    if self.args.memory_type == 'CSTU':
        feats, _ = self.mem.get_memory()
    else:
        feats, _, _ = self.mem.get_memory()

    feats = torch.stack(feats).to(x.device)
    #print('feats',feats.shape)
    dataset = torch.utils.data.TensorDataset(feats)
    data_loader = DataLoader(dataset, batch_size=self.args.inner_batch_size,
                             shuffle=True, drop_last=False, pin_memory=False)

    for e in range(self.args.epoch):
        for batch_idx, (feats_batch,) in enumerate(data_loader):
            feats_batch = feats_batch.to(x.device)

            preds, _ = self.model(feats_batch.unsqueeze(1))
            loss = self.entropy_loss(preds)

            self.model.zero_grad()
            loss.backward()

            if hasattr(self.optimizer, "first_step"):
                #print('使用 SAM 优化器')
                self.optimizer.first_step(zero_grad=True)

                # forward-backward for the second step
                preds2, _ = self.model(feats_batch.unsqueeze(1))
                loss2 = self.entropy_loss(preds2)
                self.model.zero_grad()
                loss2.backward()

                self.optimizer.second_step()
            else:
                self.optimizer.step()


    with torch.no_grad():
        self.model.eval()
        output,_ = self.model(x)
        return output, 0
def collect_params(model, freeze_top=False):

    return sam_collect_params(model, freeze_top=freeze_top)

def configure_model(model, args):
    """
    配置模型：冻结除 BN / LN / IN 层以外的所有参数，
    并设置 BN 的运行模式（track_running_stats）以及动量。
    """
    model.train()
    model.requires_grad_(False)  # 冻结全部参数

    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            if args.use_learned_stats:
                module.track_running_stats = True
                module.momentum = args.bn_momentum
            else:
                module.track_running_stats = False
                module.running_mean = None
                module.running_var = None

            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)

        elif isinstance(module, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.LayerNorm)):
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)

    return model


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)






class HLoss(nn.Module):
    def __init__(self, temp_factor=1.0):
        super(HLoss, self).__init__()
        self.temp_factor = temp_factor

    def forward(self, x):
        softmax = F.softmax(x / self.temp_factor, dim=1)
        entropy = -softmax * torch.log(softmax + 1e-6)
        b = entropy.sum(dim=1).mean()

        return b

