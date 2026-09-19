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
        self.optimizer = optimizer  # LAME does not use optimizer, but kept for a unified format

        self.force_symmetry = args.FORCE_SYMMETRY
        self.affinity = eval(f'{args.AFFINITY}_affinity')(
            sigma=args.SIGMA,
            knn=args.KNN
        )


        # Split the model structure
        self.feature_extractor = self.model
        self.classifier = get_classifier(self.args, self.model)

        # Copy the model state for reset
        self.model_state = deepcopy(self.model.state_dict())

        self.configure_model()

    def forward(self, x):
        return self.forward_and_adapt(x)

    @torch.no_grad()
    def forward_and_adapt(self, x):
        """
        Forward pass that performs the label-propagation-based adaptation process
        Uses the feature affinity matrix and Laplacian optimization to achieve test-time adaptation
        
        Args:
            x (torch.Tensor): Input test image tensor, usually of shape [B, C, H, W]
            
        Returns:
            torch.Tensor: Class probability distribution after label propagation optimization, of shape [B, num_classes]
        """
        # Rename the input to a more descriptive variable, explicitly denoting test images
        imgs_test = x

        # Extract features from the test images, ignoring the first return value of the feature extractor
        # Assume feature_extractor returns (logits, features) or a similar format
        _ ,features = self.feature_extractor(imgs_test)
        
        # Use the classifier to perform initial classification on the extracted features, obtaining the raw output
        outputs = self.classifier(features)

        # Compute the unary potential, used for label propagation
        # Formula: -log(softmax probability + small constant), 1e-10 added to avoid numerical instability
        unary = - torch.log(outputs.softmax(dim=1) + 1e-10)

        # Build the affinity kernel matrix (similarity matrix)
        # L2-normalize the features to ensure similarity computation is not affected by feature scale
        features = F.normalize(features, p=2, dim=-1)
        # Compute the affinity matrix (similarity matrix) based on the normalized features
        kernel = self.affinity(features)
        # If symmetry must be enforced, do it by averaging the kernel matrix with its transpose
        if self.force_symmetry:
            kernel = 0.5 * (kernel + kernel.t())

        # Label propagation optimization: update the output via the Laplacian optimization algorithm
        # Combine the unary potential and the affinity kernel matrix for label propagation to obtain the optimized classification result
        outputs = laplacian_optimization(unary, kernel)
        
        # Return the output after adaptation
        return outputs

    def configure_model(self):
        self.model.eval()
        self.model.requires_grad_(False)

    def reset(self):
        self.model.load_state_dict(self.model_state, strict=True)


def laplacian_optimization(unary, kernel, bound_lambda=1, max_steps=100):
    """
    Label propagation optimization algorithm based on Laplacian energy minimization
    Iteratively updates the classification probability distribution to minimize an energy function that contains a unary potential, a pairwise potential and an entropy term
    
    Args:
        unary (torch.Tensor): Unary potential matrix, of shape [N, K], where N is the number of samples and K is the number of classes
        kernel (torch.Tensor): Affinity similarity matrix, of shape [N, N], representing the similarity between samples
        bound_lambda (float): Weight coefficient of the pairwise potential, controlling how much the pairwise term affects the energy function
        max_steps (int): Maximum number of iterations, preventing the optimization process from diverging
        
    Returns:
        torch.Tensor: Optimized classification probability distribution, of shape [N, K]
    """
    # Store the energy value of each iteration, used to monitor the convergence process
    E_list = []
    # Initialize the previous energy value to infinity, used for the convergence check
    oldE = float('inf')
    # Initialize the probability distribution Y: convert the negative log of the unary potential into initial probabilities (softmax normalization)
    Y = (-unary).softmax(-1)  # [N, K]
    
    # Iterative optimization process
    for i in range(max_steps):
        # Compute the pairwise potential: bound_lambda * similarity matrix * current probability distribution
        pairwise = bound_lambda * kernel.matmul(Y)  # [N, K]
        # Exponent term: -unary potential + pairwise potential (combining the influence of both potentials)
        exponent = -unary + pairwise
        # Update the probability distribution Y: apply softmax normalization to the exponent term
        Y = exponent.softmax(-1)
        # Compute and store the current energy value (calls entropy_energy to compute the overall energy)
        E = entropy_energy(Y, unary, pairwise, bound_lambda).item()
        E_list.append(E)

        # Convergence check: stop iterating when the iteration count > 1 and the energy change is below the threshold (relative change < 1e-8)
        if (i > 1 and (abs(E - oldE) <= 1e-8 * abs(oldE))):
            # logger.info(f'Converged in {i} iterations')  # convergence log (currently commented out)
            break
        else:
            # Update the previous energy value, preparing for the next iteration
            oldE = E

    # Return the optimized probability distribution
    return Y

def entropy_energy(Y, unary, pairwise, bound_lambda):
    """
    Compute the Laplacian energy function, taking the unary potential, the pairwise potential and the entropy regularization term into account
    
    Energy function formula: E = sum( unary*Y - bound_lambda*pairwise*Y + Y*log(Y) )
    where:
    - unary*Y: unary potential term, penalizing distributions inconsistent with the initial classification
    - bound_lambda*pairwise*Y: pairwise potential term, encouraging similar samples to have similar distributions
    - Y*log(Y): entropy regularization term, encouraging smoothness of the distribution
    
    Args:
        Y (torch.Tensor): Current classification probability distribution, of shape [N, K]
        unary (torch.Tensor): Unary potential matrix, of shape [N, K]
        pairwise (torch.Tensor): Pairwise potential matrix, of shape [N, K]
        bound_lambda (float): Weight coefficient of the pairwise potential
        
    Returns:
        torch.Tensor: Scalar energy value
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
