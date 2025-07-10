import torch

class AllenCahnEnergy:
    def __init__(self, M: torch.Tensor, K: torch.Tensor, eps: float):
        """
        M, K: sparse (N×N) PyTorch tensors
        eps : interface thickness
        """
        self.M = M
        self.K = K
        self.eps = eps

    def value(self, u: torch.Tensor) -> torch.Tensor:
        # Dirichlet part
        dirich = 0.5 * self.eps * (u @ (self.K @ u))
        # Potential part: w = 1 - u^2
        w = 1 - u**2
        pot = (w @ (self.M @ w)) / (4 * self.eps)
        return dirich + pot

    def gradient(self, u: torch.Tensor) -> torch.Tensor:
        # ∇_u [ .5 eps u^T K u ] = eps * K u
        grad_dir = self.eps * (self.K @ u)
        # ∇_u pot:  (1/4eps) * 2 M w * (-2u) = -(M w * u)/eps
        w = 1 - u**2
        grad_pot = -(self.M @ w) * (2*u) / (4*self.eps)
        return grad_dir + grad_pot

    # you can add hessian-vector or full hessian if needed
