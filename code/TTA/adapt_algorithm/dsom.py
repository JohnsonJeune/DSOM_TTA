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
        self.pos_cache = {}  # newly added member variable
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
        self.classifier.weight = self.classifier.weight  # weight tensor containing gradient information
        self.num_classes = get_num_classes(args)  # get the total number of classes
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

            # select all prototypes predicted as i
            supports_i = self.warmup_supports[mask]              # shape: [K, D]
            ents_i = self.warmup_ent[mask] 
            prob_map = self.warmup_prob_map[mask]                      # shape: [K, 1]
    
            # normalize supports
            supports_i = F.normalize(supports_i, dim=1)

            # store into the cache or your later initialization structure (e.g., positive sample cache)
            for s, e, p in zip(supports_i, ents_i, prob_map):
                self.pos_cache[i] = []
                self.pos_cache[i].append([s.unsqueeze(0), e.item(), p.unsqueeze(0) ])

        
            self.thresh = self.args.thresh

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
        # determine the item to add to the cache based on whether the probability map is included
        item = features_loss if not include_prob_map else features_loss[:2] + [features_loss[2]]
        # check whether the predicted class already exists in the cache
        if pred in cache:
            # if the number of cached items of this class is less than the maximum capacity
            if len(cache[pred]) < shot_capacity:
                # directly append the new item to the cache list of this class
                cache[pred].append(item)
            # if the loss of the new item is smaller than the loss of the last item (the largest loss) in the cache list of this class
            elif features_loss[1] < cache[pred][-1][1] - 1e-6:
                # replace the last item with the new item
                cache[pred][-1] = item
            # sort the cache list of this class by loss value so that items with smaller loss come first
            cache[pred] = sorted(cache[pred], key=operator.itemgetter(1))
        else:
            # if the predicted class is not in the cache, create a new cache list for this class and append the new item
            cache[pred] = [item]

def compute_cache_logits(image_features, cache, alpha, beta, weights, neg_mask_thresholds=None):
    """Compute logits using positive/negative cache with numerical stability."""
    # initialize lists for storing the cached features (keys)
    cache_keys = []
    cache_values = []
    prob_map = []

    # iterate over the cache and aggregate prototypes by class
    for class_index in sorted(cache.keys()):
        proto_list = cache[class_index]

        # stack the probability maps item[2] of all prototypes of the current class
        proto_probs = [item[2] for item in proto_list]
        proto_probs = torch.stack(proto_probs, dim=0)  # [num_proto, ...]
        proto_mean = proto_probs.mean(dim=0)  # average over prototypes

        # store the mean probability map of the current class into prob_map
        prob_map.append(proto_mean)

        # store features and labels
        for item in proto_list:
            cache_keys.append(item[0])
            if neg_mask_thresholds:
                cache_values.append(item[2])  # original prob
            else:
                cache_values.append(class_index)

    # concatenate all feature tensors in the cache_keys list along dim 0
    cache_keys = torch.cat(cache_keys, dim=0).permute(1, 0)
    prob_map = torch.cat(prob_map, dim=0)
    #print(prob_map.shape)


    # if negative sample mask thresholds are provided
    if neg_mask_thresholds:
        # concatenate all mask value tensors in the cache_values list along dim 0
        cache_values = torch.cat(cache_values, dim=0)
        # create a boolean mask indicating whether the mask values are within the threshold range
        cache_values = (((cache_values > neg_mask_thresholds[0]) & (cache_values < neg_mask_thresholds[1])).type(torch.int8)).half()
    else:
        # otherwise, convert the class indices into a one-hot encoded tensor
        cache_values = F.one_hot(torch.tensor(cache_values, dtype=torch.int64).to(cache_keys.device), num_classes=weights.size(0)).to(torch.float32)

    # ensure cache_values is of type float32 to match the type of subsequent computations
    cache_values = cache_values.float()
    
    # compute the affinity matrix (cosine similarity) between image features and cached features
    image_features_n = F.normalize(image_features, dim=1)  # [B, D] normalized row-wise
    cache_keys_n = F.normalize(cache_keys, dim=0)          # [D, K] normalized column-wise
    affinity = (image_features_n @ cache_keys_n)
    cache_logits = ((-1) * (beta - beta * affinity)).exp() @ cache_values
    return alpha * cache_logits, prob_map

def get_logits(x, model, weights, classifier):
    images = x
    output = model(images)
    if isinstance(output, tuple):
        # unpack the model output (assumed to be a (logits, features) tuple)
        _, feature = output
        # compute the class probability distribution through the classifier
        logits = classifier(feature)
        ent = softmax_entropy(logits)
        prob_map = torch.softmax(logits, dim=1)
        pred = logits.topk(1, 1, True, True)[1].squeeze(1)
        feature = torch.nn.functional.normalize(feature, dim=1)
    else:
        # directly return the model output in non-tuple form
        feature = output
        logits = classifier(feature)
        ent = softmax_entropy(logits)
        prob_map = torch.softmax(logits, dim=1)
        pred = logits.topk(1, 1, True, True)[1].squeeze(1)
        feature = torch.nn.functional.normalize(feature, dim=1)
    return feature, logits, ent, prob_map, pred

@torch.enable_grad() 
def run_test_dsom(self, pos_cfg, neg_cfg, x, model, weights, classifier, pos_cache, neg_cache):
    steps = 1
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

        loss_hard = nn.CrossEntropyLoss()(logits_pasle[mask_hard], logits_pasle[mask_hard].detach().argmax(1))
        loss_partial = cc_loss(logits_pasle[mask_partial], partial_labels.detach())
        lam_hard = sum(mask_hard.long()) / (sum(mask_hard.long()) + sum(mask_partial.long()))
        loss1 = loss_hard * lam_hard + loss_partial * (1 - lam_hard)



        loss1.backward()
        self.optimizer.step()

    with torch.no_grad():
        # unpack the configuration parameters
        pos_enabled, neg_enabled = pos_cfg['enabled'], neg_cfg['enabled']
        pos_params = pos_cfg if pos_enabled else None
        neg_params = neg_cfg if neg_enabled else None
        num_classes = classifier.weight.size(0)  # number of classes of the classifier weight

        # 1. obtain the features and logits of the whole batch at once
        image_features, logits, ent, prob_map, pred = get_logits(x, model, weights, classifier)


        batch_size = x.size(0)
        # 2. update the cache in batch
        if pos_enabled:
            for idx in range(batch_size):
                single_pred = pred[idx]
                single_ent = ent[idx].item()
                single_feature = image_features[idx].unsqueeze(0)
                single_prob_map = prob_map[idx].unsqueeze(0)
                update_cache(pos_cache, single_pred, [single_feature, single_ent, single_prob_map], pos_params['shot_capacity'], include_prob_map=True)

        # 3. compute the final logits in batch
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
    # initialize residual
    residual = nn.Parameter(torch.zeros([N, K], device=unary.device))


    lr =6.
    max_steps = 1

    optimizer = torch.optim.Adam([residual], lr=lr)

    bound_lambda = 1.


    oldE = float("inf")
    for i in range(max_steps):
        

        optimizer.zero_grad()


        # adjust logits
        logit_ = -unary + residual  # [N, K]
        Y = F.softmax(logit_, dim=-1)  # probability distribution [N, K]
        # --- energy function part (formula from the paper) ---
        energy = -torch.logsumexp(logit_, dim=-1)   # [N]
        energy_loss = energy.mean()

        # --- pairwise regularization term ---
        pairwise = kernel @ Y   # [N, K]
        pairwise_loss = -bound_lambda * (Y * pairwise).sum() / N

        # --- total loss ---
        loss = energy_loss + pairwise_loss

        loss.backward()
        optimizer.step()


    # final output Y (detach to avoid gradient accumulation)
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


def entropy_energy(Y, unary, pairwise, bound_lambda):
    """
    Compute the energy function E(Y)
    Y: [N, K] probability distribution
    unary: [N, K] unary cost
    pairwise: [N, K] adjacency regularization term
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

def cc_loss(outputs, partialY):
    sm_outputs = F.softmax(outputs, dim=1)
    final_outputs = sm_outputs * partialY
    average_loss = - torch.log(final_outputs.sum(dim=1)).mean()
    return average_loss
