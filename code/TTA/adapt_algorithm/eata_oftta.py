# encoding=utf-8
"""EATA (Efficient Anti-forgetting Test-time Adaptation) adapted to OFTTA CNN models.
Extracted from EATA-main/eata.py, wrapped for the OFTTA interface (args, model) -> module with forward(x)->logits.
Adapts BatchNorm affine only; entropy minimization with reliability+redundancy sample selection;
optional EWC anti-forgetting (disabled unless fishers provided).
"""
from copy import deepcopy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np_, p in m.named_parameters():
                if np_ in ['weight', 'bias']:
                    params.append(p); names.append(f'{nm}.{np_}')
    return params, names


def copy_model_and_optimizer(model, optimizer):
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


@torch.jit.script
def update_model_probs(current_model_probs, probs):
    return 0.9 * current_model_probs + (1 - 0.9) * probs


class EATA(nn.Module):
    def __init__(self, args, model, steps=1, episodic=False, e_margin=math.log(1000)/2-1,
                 d_margin=0.05, fisher_alpha=2000.0, fishers=None):
        super().__init__()
        self.model = configure_model(model)
        self.args = args
        self.steps = steps
        self.episodic = episodic
        self.e_margin = e_margin
        self.d_margin = d_margin
        self.fisher_alpha = fisher_alpha
        self.fishers = fishers if fishers is not None else {}
        self.num_samples_update = 0
        self.current_model_probs = None
        self.params, self.param_names = collect_params(self.model)
        self.optimizer = optim.Adam(self.params, lr=args.lr if hasattr(args, 'lr') else 1e-3)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(self.model, self.optimizer)
        self.n_kept_ent = 0

    def forward(self, x):
        if self.episodic:
            load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        for _ in range(self.steps):
            outputs, self.current_model_probs = forward_and_adapt(
                x, self.model, self.optimizer, self.fishers, self.e_margin,
                self.current_model_probs, self.fisher_alpha, self.d_margin)
        return outputs

    def reset(self):
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)


@torch.enable_grad()
def forward_and_adapt(x, model, optimizer, fishers, e_margin, current_model_probs,
                      fisher_alpha, d_margin):
    outputs = model(x)
    if isinstance(outputs, (tuple, list)):
        outputs = outputs[0]
    entropys = softmax_entropy(outputs)
    probs = outputs.softmax(1)

    # 1. reliability filter: keep low-entropy samples
    filter_id_1 = entropys < e_margin
    if filter_id_1.sum() == 0:
        return outputs, current_model_probs
    entropys = entropys[filter_id_1]
    outputs_filter = outputs[filter_id_1]
    probs = probs[filter_id_1]
    n_after1 = entropys.shape[0]

    # 2. redundancy filter: drop samples close to current moving avg
    if current_model_probs is not None:
        filter_id_2 = 1 - (probs @ current_model_probs) > d_margin
        entropys = entropys[filter_id_2]
        outputs_filter = outputs_filter[filter_id_2]
        probs = probs[filter_id_2]
        n_after2 = entropys.shape[0]
    else:
        n_after2 = n_after1
    if entropys.shape[0] == 0:
        return outputs, current_model_probs

    # reweighted entropy
    coeff = 1 / torch.exp(entropys - e_margin)
    loss = (entropys * coeff).mean(0)

    # EWC anti-forgetting (only if fishers provided; by default fishers={} so skipped)
    if fishers is not None and len(fishers) > 0:
        ewc_loss = 0
        params, names = collect_params(model)
        for name, param in zip(names, params):
            if name in fishers:
                fisher, fparam = fishers[name]
                ewc_loss += torch.sum(fisher * (param - fparam) ** 2)
        loss += fisher_alpha * ewc_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # update moving avg
    if current_model_probs is not None:
        current_model_probs = update_model_probs(current_model_probs, probs.mean(0))
    else:
        current_model_probs = probs.mean(0)
    return outputs, current_model_probs
