from copy import deepcopy
import numpy as np
import operator
import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
import torch.nn.functional as F

def get_classifier(args, model):
    return model.classifier



class dsom(nn.Module):
    """Tent adapts a model by entropy minimization during testing.

    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, args, model, steps=1, episodic=False):
        super().__init__()
        self.args = args
        self.pos_cache = {}  # 新增成员变量
        self.neg_cache = {}
        self.cfg =args
        self.model = model
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic
        self.force_symmetry = args.FORCE_SYMMETRY
        self.affinity = eval(f'{args.AFFINITY}_affinity')(
            sigma=args.SIGMA,
            knn=args.KNN
        )
        self.classifier = get_classifier(self.args, self.model)
        self.classifier.weight = self.classifier.weight  # 包含梯度信息的权重张量
        self.num_classes = get_num_classes(args)  # 获取类别总数
        #self.cfg = get_config_file(self.args.config_path)
        self.warmup_supports = self.classifier.weight.data
        warmup_logits = self.classifier(self.classifier.weight.data)
        self.warmup_ent = softmax_entropy(warmup_logits).unsqueeze(1)
        self.warmup_prob_map = torch.softmax(warmup_logits, dim=1)
        self.warmup_pred = warmup_logits.topk(1, 1, True, True)[1].squeeze(1)


        for i in range(self.num_classes):


            mask = self.warmup_pred == i
            if not mask.any():
                self.pos_cache[i] = []

            # 选择所有预测为 i 的 prototype
            supports_i = self.warmup_supports[mask]              # shape: [K, D]
            ents_i = self.warmup_ent[mask] 
            prob_map = self.warmup_prob_map[mask]                      # shape: [K, 1]
    
            # 对 supports 做归一化
            supports_i = F.normalize(supports_i, dim=1)

            # 存入 cache 或你后续的初始化结构（比如正样本缓存）
            for s, e, p in zip(supports_i, ents_i, prob_map):
                self.pos_cache[i] = []
                self.pos_cache[i].append([s.unsqueeze(0), e.item(), p.unsqueeze(0) ])

        
            self.thresh = self.args.thresh
            self.thresh_end = self.args.thresh - self.args.thresh_gap
            self.thresh_des = self.args.thresh_des
            self.temp = self.args.temp
            self.buffer_size = self.args.buffer_size

            self.samples_buffer = None


        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.classifier = get_classifier(self.args, self.model)
        self.classifier.weight = self.classifier.weight 
        self.model = configure_model(self.model)
        params, param_names = collect_params(self.model)
        self.optimizer = setup_optimizer(args, params )
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()

        for _ in range(self.steps):
            outputs, self.pos_cache, self.neg_cache = run_test_dsom(self, self.cfg.positive, self.cfg.negative, x, self.model, self.classifier.weight,self.classifier,
            self.pos_cache, self.neg_cache)
        return outputs
    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)




@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)




def collect_params(model):
    """Collect the affine scale + shift parameters from batch norms.
    Walk the model's modules and collect all batch normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names



def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state


def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def configure_model(model):
    """Configure model for use with tent."""
    model.train()
    model.requires_grad_(False)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
            m.track_running_stats = False
            # m.running_mean = None
            # m.running_var = None
    return model

def setup_optimizer(args, params):
    if True:
        return optim.Adam(params,
                    lr=1e-2,
                    )
    else:
        raise NotImplementedError
    

def update_cache(cache, pred, features_loss, shot_capacity, include_prob_map=False):
    pred = pred.item()
    with torch.no_grad():
        # 根据是否包含概率图来确定要添加到缓存中的项
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        # 检查预测类别是否已经存在于缓存中
        if pred in cache:
            # 如果该类别缓存项数量小于最大容量
            if len(cache[pred]) < shot_capacity:
                # 直接将新项添加到该类别的缓存列表中
                cache[pred].append(item)
            # 如果新项的损失小于该类别缓存列表中最后一项（损失最大项）的损失
            elif features_loss[1] < cache[pred][-1][1] - 1e-6:
                # 用新项替换最后一项
                cache[pred][-1] = item
            # 对该类别缓存列表按损失值进行排序，确保损失小的项在前
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            # 如果预测类别不在缓存中，为该类别创建一个新的缓存列表，并将新项添加进去
            cache[pred] = [item]

def compute_cache_logits(image_features, cache, alpha, beta, weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache with numerical stability."""
    # 初始化列表，用于存储缓存的特征（键）
    cache_keys = []
    cache_values = []
    prob_map = []

    # 遍历缓存，按类别聚合 proto
    for class_index in sorted(cache.keys()):
        proto_list = cache[class_index]

        # 将当前类所有 proto 的概率图 item[2] 堆叠起来
        proto_probs = [item[2] for item in proto_list]
        proto_probs = torch.stack(proto_probs, dim=0)  # [num_proto, ...]
        proto_mean = proto_probs.mean(dim=0)  # 对 proto 求平均

        # 将当前类的平均概率图存到 prob_map
        prob_map.append(proto_mean)

        # 存储特征和标签
        for item in proto_list:
            cache_keys.append(item[0])
            if neg_mask_thresholds:
                cache_values.append(item[2])  # 原始 prob
            else:
                cache_values.append(class_index)

    # 将 cache_keys 列表中的所有特征张量在第 0 维拼接
    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
    prob_map = torch.cat(prob_map, dim=0)
    #print(prob_map.shape)


    # 如果提供了负样本掩码阈值
    if neg_mask_thresholds:
        # 将 cache_values 列表中的所有掩码值张量在第 0 维拼接
        cache_values = torch.cat(cache_values, dim=0)
        # 创建一个布尔掩码，判断掩码值是否在阈值范围内
        cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).half()
    else:
        # 否则，将类别索引转换为 one-hot 编码张量
        cache_values = F.one_hot(torch.tensor(cache_values, dtype=torch.int64).to(cache_keys.device), num_classes=weights.size(0)).to(torch.float32)

    # 确保 cache_values 为 float32 类型，以匹配后续计算的类型
    cache_values = cache_values.float()
    
    # 计算图像特征与缓存特征的亲和力矩阵（余弦相似度）
    image_features_n = F.normalize(image_features, dim=1)  # [B, D] 按行归一
    cache_keys_n = F.normalize(cache_keys, dim=0)          # [D, K] 按列归一
    affinity = (image_features_n @ cache_keys_n)
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits, prob_map

def get_logits(x, model, weights, classifier):
    images = x
    output = model(images)
    if isinstance(output, tuple):
        # 解构模型输出（假设为(logits, features)元组）
        _, feature = output
        # 通过分类器计算类别概率分布
        logits = classifier(feature)
        ent = softmax_entropy(logits)
        prob_map = torch.softmax(logits, dim=1)
        pred = logits.topk(1, 1, True, True)[1].squeeze(1)
        feature = torch.nn.functional.normalize(feature, dim=1)
    else:
        # 直接返回非元组形式的模型输出
        feature = output
        logits = classifier(feature)
        ent = softmax_entropy(logits)
        prob_map = torch.softmax(logits, dim=1)
        pred = logits.topk(1, 1, True, True)[1].squeeze(1)
        feature = torch.nn.functional.normalize(feature, dim=1)
    return feature, logits, ent, prob_map, pred

@torch.enable_grad() 
def run_test_dsom(self, pos_cfg, neg_cfg, x, model, weights, classifier, pos_cache, neg_cache):
    steps = 2
    for i in range(steps):
        self.optimizer.zero_grad()

        if self.samples_buffer != None:
            x_pasle = torch.cat((x,self.samples_buffer),dim=0)
        else:
            x_pasle =x

        logits_pasle,_ = self.model(x_pasle)
        probs = F.softmax(logits_pasle,1)
        probs_des, _ = torch.sort(probs, descending=True)
        margins = probs_des[:,0] - probs_des[:,1]
        mask_hard = margins > self.thresh
        mask_partial = ~ (mask_hard )






        partial_labels = ((probs[mask_partial] + self.thresh) > probs_des[mask_partial][:,0].reshape(-1,1)).long()

        loss_hard = nn.CrossEntropyLoss()(logits_pasle[mask_hard] / self.temp, logits_pasle[mask_hard].detach().argmax(1))
        loss_partial = cc_loss(logits_pasle[mask_partial], partial_labels.detach(), self.temp)
        lam_hard = sum(mask_hard.long()) / (sum(mask_hard.long()) + sum(mask_partial.long()))
        loss1 = loss_hard * lam_hard + loss_partial * (1 - lam_hard)



        loss1.backward()
        self.optimizer.step()
        if self.thresh > self.thresh_end:
            self.thresh -= self.thresh_des

    with torch.no_grad():
        # 解包配置参数
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        pos_params = pos_cfg if pos_enabled else None
        neg_params = neg_cfg if neg_enabled else None
        num_classes = classifier.weight.size(0)  # 分类器权重的类别数

        # 1. 一次性获取整个批次的特征和logits
        image_features, logits, ent, prob_map, pred = get_logits(x, model, weights, classifier)


        batch_size = x.size(0)
        # 2. 批量更新缓存
        if pos_enabled:
            for idx in range(batch_size):
                single_pred = pred[idx]
                single_ent = ent[idx].item()
                single_feature = image_features[idx].unsqueeze(0)
                single_prob_map = prob_map[idx].unsqueeze(0)
                update_cache(pos_cache, single_pred, [single_feature, single_ent, single_prob_map], pos_params['shot_capacity'], include_prob_map=True)

        # 3. 批量计算最终的logits
        final_logits = logits.clone()

        if pos_enabled and pos_cache:
            pos_cache_logits, proto_map = compute_cache_logits(image_features, pos_cache, pos_params['alpha'], pos_params['beta'], weights)
            final_logits += pos_cache_logits
    unary = - torch.log(final_logits.softmax(dim=1) + 1e-10).detach()
    features = F.normalize(image_features, p=2, dim=-1)
    kernel = self.affinity(features)
    kernel =kernel.detach()
    #print(proto_map.shape)
    if self.force_symmetry:
        kernel = 0.5 * (kernel + kernel.t())
    N, K = unary.shape
    # 初始化 residual
    residual = nn.Parameter(torch.zeros([N, K], device=unary.device))


    lr =6.
    max_steps = 1

    optimizer = torch.optim.Adam([residual], lr=lr)

    bound_lambda = 1.


    oldE = float("inf")
    for i in range(max_steps):
        

        optimizer.zero_grad()


        # 调整 logits
        logit_ = -unary + residual  # [N, K]
        Y = F.softmax(logit_, dim=-1)  # 概率分布 [N, K]
        # --- 能量函数部分 (论文公式) ---
        energy = -torch.logsumexp(logit_, dim=-1)   # [N]
        energy_loss = energy.mean()

        # --- pairwise 正则项 ---
        pairwise = kernel @ Y   # [N, K]
        pairwise_loss = -bound_lambda * (Y * pairwise).sum() / N

        # --- 总 loss ---
        loss = energy_loss + pairwise_loss

        loss.backward()
        optimizer.step()


    # 最终输出 Y（detach 避免梯度累积）
    Y = torch.softmax(-unary + residual, dim=-1).detach()
    return Y, pos_cache, neg_cache

    
def get_num_classes(args):
    print(args.dataset, 'args.dataset')
    if args.dataset == "uci":
        num_classes = 6
    elif args.dataset == 'unimib':
        num_classes = 17
    elif args.dataset == 'oppo':
        num_classes = 17
    elif args.dataset == 'pamap2':
        num_classes = 12
    elif args.dataset == 'usc':
        num_classes = 12
    else:
        print('not this dataset')

    return num_classes

import torch
import torch.nn.functional as F

def infoNCE_loss(sample_probs, proto_prob, reduction="mean"):
    """
    sample_probs: Tensor [B, D]  - B 个样本
    proto_prob:   Tensor [K, D]  - K 个原型
    reduction: "mean" or "none"
    """
    # 归一化到单位球面（余弦相似度）
    sample_norm = F.normalize(sample_probs, p=2, dim=-1)  # [B, D]
    proto_norm = F.normalize(proto_prob, p=2, dim=-1)     # [K, D]

    # 相似度矩阵 [B, K]
    sim_matrix = torch.matmul(sample_norm, proto_norm.t())

    # 找到每个样本最近的原型索引
    k_star = torch.argmax(sim_matrix, dim=1)  # [B]

    # softmax over prototypes
    log_probs = F.log_softmax(sim_matrix, dim=1)  # [B, K]

    # 取正例 log prob
    loss = -log_probs[torch.arange(sim_matrix.size(0)), k_star]  # [B]

    if reduction == "mean":
        return loss.mean()
    else:
        return loss

def entropy_energy(Y, unary, pairwise, bound_lambda):
    """
    计算能量函数 E(Y)
    Y: [N, K] 概率分布
    unary: [N, K] 单点代价
    pairwise: [N, K] 邻接正则项
    """

    E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clamp(min=1e-20))).sum()
    #E = (unary * Y - bound_lambda * pairwise * Y + Y * torch.log(Y.clamp(min=1e-20))).sum()
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

def cc_loss(outputs, partialY, temp):
    sm_outputs = F.softmax(outputs / temp, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss