import torch

def hessian_dense(E_grad: torch.tensor, u: torch.tensor, edge_list: torch.tensor):
        """
        Dense Hessian via one batched autograd cal

        Returns : Hessian 2d tensor
        Cost    : O(n**2)
        """
        N = u.numel()
        I = torch.eye(N, dtype=u.dtype, device=u.device)
        H = torch.autograd.grad(E_grad, u,
                                grad_outputs=I,
                                retain_graph=True,
                                is_grads_batched=True)[0]   # (N,N)                            
        
        return H