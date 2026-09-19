# encoding=utf-8
"""RoTTA (Robust Test-Time Adaptation) simplified, adapted to OFTTA CNN models.
Extracted from RoTTA-main/core/adapter/rotta.py. Simplified for the OFTTA interface
(args, model) -> module with forward(x)->logits. Keeps the core ideas:
  * student model (BN affine trainable) + EMA teacher
  * reliability-weighted memory bank (sample timeliness = sigmoid(-age))
  * soft pseudo-label distribution matching student<->EMA teacher
  * EMA teacher update
Dropped (framework-incompatible): yacs cfg, strong-augmentation transforms, custom RobustBN layers.
"""
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MEMORY_SIZE = 64
UPDATE_FREQUENCY = 64
NU = 0.001   # EMA teacher update factor


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def configure_model(model):
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            if m.running_mean is not None:
                m.running_mean = None
                m.running_var = None
    return model


def collect_params(model):
    params = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            params.extend([p for p in m.parameters() if p.requires_grad])
    return params


class RoTTA(nn.Module):
    def __init__(self, args, model, steps=1, episodic=False):
        super().__init__()
        self.model = configure_model(model)          # student
        self.model_ema = deepcopy(self.model)        # EMA teacher
        for p in self.model_ema.parameters():
            p.requires_grad_(False)
        self.model_ema.eval()
        self.steps = steps
        self.episodic = episodic
        self.lr = args.lr if hasattr(args, 'lr') else 1e-3
        self.params = collect_params(self.model)
        self.optimizer = optim.Adam(self.params, lr=self.lr)
        # memory bank: list of (data, teacher_softmax, age)
        self.memory = []
        self.model_state = deepcopy(self.model.state_dict())
        self.optimizer_state = deepcopy(self.optimizer.state_dict())

    def forward(self, x):
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            outputs = forward_and_adapt(x, self.model, self.model_ema, self.optimizer,
                                        self.memory, NU)
        return outputs

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)
        self.optimizer.load_state_dict(self.optimizer_state)
        self.memory = []


@torch.enable_grad()
def forward_and_adapt(x, model, model_ema, optimizer, memory, nu):
    # EMA teacher forward (no grad)
    with torch.no_grad():
        ema_out = model_ema(x)
        if isinstance(ema_out, (tuple, list)):
            ema_out = ema_out[0]
        teacher_soft = ema_out.softmax(1).detach()
        ent = softmax_entropy(ema_out).detach()

    # store into memory bank with age 0
    for i in range(x.size(0)):
        memory.append((x[i:i+1].detach(), teacher_soft[i:i+1], 0))
        if len(memory) > MEMORY_SIZE:
            memory.pop(0)
    # increment ages
    for i in range(len(memory)):
        memory[i] = (memory[i][0], memory[i][1], memory[i][2] + 1)

    # periodically update student on memory (timeliness reweighted distribution matching)
    if len(memory) >= UPDATE_FREQUENCY:
        xs = torch.cat([item[0] for item in memory], dim=0)
        soft_t = torch.cat([item[1] for item in memory], dim=0)
        ages = torch.tensor([item[2] for item in memory], dtype=torch.float32, device=x.device)
        w = torch.sigmoid(-ages)   # down-weight stale samples
        stu_out = model(xs)
        if isinstance(stu_out, (tuple, list)):
            stu_out = stu_out[0]
        loss = -(soft_t * F.log_softmax(stu_out, dim=1)).sum(1) * w
        loss = loss.mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # EMA update teacher
        with torch.no_grad():
            for tp, sp in zip(model_ema.parameters(), model.parameters()):
                tp.data.mul_(1 - nu).add_(sp.data, alpha=nu)
        memory.clear()
    return ema_out
