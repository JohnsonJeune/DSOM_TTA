from copy import deepcopy
import os
import torch
import torch.jit
import torch.nn as nn
import torch.nn.functional as F
from device import DEVICE


def init_random(x):
    return torch.empty_like(x).uniform_(-1, 1)


class EnergyModel(nn.Module):
    def __init__(self, model):
        super(EnergyModel, self).__init__()
        self.f = model

    def classify(self, x):
        penult_z= self.f(x)
        return penult_z
    
    def forward(self, x, y=None):
        logits,_ = self.classify(x)
        if y is None:
            return logits.logsumexp(1), logits
        else:
            return torch.gather(logits, 1, y[:, None]), logits
        

def sample_p_0(x, reinit_freq, replay_buffer, bs, im_sz, n_ch, device, y=None):
    if len(replay_buffer) == 0:
        return init_random(x), []
    buffer_size = len(replay_buffer)
    inds = torch.randint(0, buffer_size, (bs,))
    # if cond, convert inds to class conditional inds

    buffer_samples = replay_buffer[inds].to(DEVICE)
    random_samples = init_random(x).to(DEVICE)
    choose_random = (torch.rand(bs) < reinit_freq).float()[:, None, None, None].to(DEVICE)
    samples = choose_random * random_samples + (1 - choose_random) * buffer_samples
    return samples.to(device), inds


def sample_q(x, f, replay_buffer, n_steps, sgld_lr, sgld_std, reinit_freq, batch_size, im_sz, n_ch, device, y=None):
    """this func takes in replay_buffer now so we have the option to sample from
    scratch (i.e. replay_buffer==[]).  See test_wrn_ebm.py for example.
    """
    f.eval()
    # get batch size
    bs = batch_size if y is None else y.size(0)
    # generate initial samples and buffer inds of those samples (if buffer is used)
    init_sample, buffer_inds = sample_p_0(x, reinit_freq=reinit_freq, replay_buffer=replay_buffer, bs=bs, im_sz=im_sz, n_ch=n_ch, device=device ,y=y)
    init_samples = deepcopy(init_sample)
    x_k = torch.autograd.Variable(init_sample, requires_grad=True)
    # sgld
    for k in range(n_steps):
        f_prime = torch.autograd.grad(f(x_k, y=y)[0].sum(), [x_k], retain_graph=True)[0]
        x_k.data += sgld_lr * f_prime + sgld_std * torch.randn_like(x_k)
    f.train()
    final_samples = x_k.detach()
    # update replay buffer
    if len(replay_buffer) > 0:
        buffer_inds = buffer_inds.to(DEVICE)
        final_samples =final_samples.to(DEVICE)
        replay_buffer = replay_buffer.to(DEVICE)
        replay_buffer[buffer_inds] = final_samples

    return final_samples.to(device), init_samples.detach()

class TEA(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, steps=1, episodic=False, 
                 buffer_size=10000, sgld_steps=20, sgld_lr=1, sgld_std=0.01, reinit_freq=0.05, if_cond='uncond', 
                 n_classes=10, im_sz=32, n_ch=3, path=None, logger=None): 
        super().__init__()

        self.energy_model=EnergyModel(model)
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        self.sgld_steps = sgld_steps
        self.sgld_lr = sgld_lr
        self.sgld_std = sgld_std
        self.reinit_freq = reinit_freq
        self.if_cond = if_cond
        
        self.n_classes = n_classes
        self.im_sz = im_sz
        self.n_ch = n_ch
        
        self.path=path
        self.logger = logger

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.energy_model, self.optimizer)

    def forward(self, x, if_adapt=True, counter=None, if_vis=False):
        self.replay_buffer = init_random(x)
        self.replay_buffer_old = deepcopy(self.replay_buffer)
        if self.episodic:
            self.reset()
        
        if if_adapt:
            for i in range(self.steps):
                outputs = forward_and_adapt(x, self.energy_model, self.optimizer, 
                                            self.replay_buffer, self.sgld_steps, self.sgld_lr, self.sgld_std, self.reinit_freq,
                                            if_cond=self.if_cond, n_classes=self.n_classes)
        else:
            self.energy_model.eval()
            with torch.no_grad():
                outputs,_ = self.energy_model.classify(x)

        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.energy_model, self.optimizer,
                                 self.model_state, self.optimizer_state)
        

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, energy_model, optimizer, replay_buffer, sgld_steps, sgld_lr, sgld_std, reinit_freq, if_cond, n_classes=10):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    batch_size=x.shape[0]
    n_ch = x.shape[1]
    im_sz = x.shape[2]
    device = x.device
    
    if if_cond == 'uncond':
        x_fake, _ = sample_q(x, energy_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=None)
    elif if_cond == 'cond':
        y = torch.randint(0, n_classes, (batch_size,)).to(device)
        x_fake, _ = sample_q(x, energy_model, replay_buffer, 
                             n_steps=sgld_steps, sgld_lr=sgld_lr, sgld_std=sgld_std, reinit_freq=reinit_freq, 
                             batch_size=batch_size, im_sz=im_sz, n_ch=n_ch, device=device, y=y)

    # forward
    out_real = energy_model(x)
    energy_real = out_real[0].mean()
    energy_fake = energy_model(x_fake)[0].mean()

    # adapt
    loss = (- (energy_real - energy_fake)) 
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    outputs = energy_model.classify(x)

    return outputs


def collect_params(model, ada_param=['bn'], logger=None):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []

    if 'all' in  ada_param:

        return model.parameters(), 'all'
    
    if 'bn' in ada_param:

        for nm, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d): 
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

    if 'gn' in ada_param:

        for nm, m in model.named_modules():
            if isinstance(m, nn.GroupNorm): 
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    
    if 'in' in ada_param:

        for nm, m in model.named_modules():
            if isinstance(m, nn.InstanceNorm2d): 
                for np, p in m.named_parameters():
                    print(np)
                    exit(0)
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")

    if 'conv' in ada_param:

        for nm, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    if 'fc' in ada_param:

        for nm, m in model.named_modules():
            if isinstance(m, nn.Linear):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:  # weight is scale, bias is shift
                        params.append(p)
                        names.append(f"{nm}.{np}")
    return params, names


def configure_model(model, ada_param=['bn']):
    """Configure model for use with tent."""

    # train mode, because tent optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what tent updates
    model.requires_grad_(False)
    
    if 'bn' in ada_param:
        # configure norm for model updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    if 'gn' in ada_param:
        # configure norm for model updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.GroupNorm):
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    
    if 'in' in ada_param:
        # configure norm for model updates: enable grad + force batch statisics
        for m in model.modules():
            if isinstance(m, nn.InstanceNorm2d): 
                m.requires_grad_(True)
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None

    if 'conv' in ada_param:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.requires_grad_(True)

    if 'fc' in ada_param:
        for m in model.modules():
            if isinstance(m, nn.Linear):
                m.requires_grad_(True)
    return model


def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)