import torch
import math

# barycentric monomial integral over simplex
def bary_integral(alpha, beta, gamma, *, dtype=None, device=None):
    """
    alpha, beta, gamma  : int, float, or torch.Tensor (broadcastable)
    dtype, device       : optional overrides when inputs are Python ints
    Return  integral of (phi_1)^a * (phi_2)^b * (phi_3)^b  
    """
    # Promote to tensor for vectorisation
    alpha = torch.as_tensor(alpha, dtype=dtype, device=device)
    beta  = torch.as_tensor(beta,  dtype=alpha.dtype, device=alpha.device)
    gamma = torch.as_tensor(gamma, dtype=alpha.dtype, device=alpha.device)

    # log-factorial(n) = lgamma(n+1)
    numer = torch.lgamma(alpha + 1) + torch.lgamma(beta + 1) + torch.lgamma(gamma + 1)
    denom = torch.lgamma(alpha + beta + gamma + 3)        # (…+2)! ⇒ +3 in lgamma

    coeff = 2.0 * torch.exp(numer - denom)                # shape = broadcast result
    return coeff 

class AllenCahnEnergy:
    """
    Assemble and evaluate the Allen-Cahn Energy
    E_e(u) = 0.5e int_Omega |grad_u|^2 + (1-u^2)^2/(4e)    
    for a P1 finite-element field on a triangular mesh.
    ----------
    area: PyTorch tensor shape (T,) of each triangles area
    basis_matrix: (T,N,N) PyTorch tensors of basis matrix for each triangle
    eps : interface thickness
    """

    def __init__(self, areas: torch.Tensor, basis_matrices: torch.Tensor, eps: float):
        self.areas = areas
        self.basis_matrices = basis_matrices
        self.eps = eps

        # degree-4 barycentric integrals (scalars on the same device / dtype)
        dev, dt = areas.device, areas.dtype
        self.I400  = bary_integral(4,0,0, dtype=dt, device=dev)   # ∫ λ_i⁴
        self.I310  = bary_integral(3,1,0, dtype=dt, device=dev)   # ∫ λ_i³ λ_j
        self.I220  = bary_integral(2,2,0, dtype=dt, device=dev)   # ∫ λ_i² λ_j²
        self.I211 = bary_integral(2,1,1, dtype=dt, device=dev)   # ∫ λ_i² λ_j λ_k
        self.I200 = bary_integral(2,0,0, dtype=dt, device=dev)   # ∫ λ_i²
        self.I110 = bary_integral(1,1,0, dtype=dt, device=dev)   # ∫ λ_i λ_j

    # ---- potential energy on each triangle ----------------------------
    def _potential_per_triangle(self, u1, u2, u3):
        """
        Compute int_triangle (1-u^2)^2 for every triangle in a vectorised way
        ----------
        Arguments are (T,) tensors.
        ----------
        Returns a (T,) tensors.
        """

        u = self.areas * ( 1
        + self.I400 * ( u1**4 + u2**4 + u3**4 ) 
        - 2 * self.I200 * ( u1**2 + u2**2 + u3**2 )
        - 4 * self.I110 * ( u1*u2 + u1*u3 + u2*u3 )
        + 6 * self.I220 * ( (u1*u2)**2 + (u1*u3)**2 + (u2*u3)**2 )
        + 4 * self.I310 * ( u1*u2**3 + u1**3*u2 + u1*u3**3 + u1**3*u3 + u2*u3**3 + u2**3*u3)
        + 12 * self.I211 * (u1**2*u2*u3 + u1*u2**2*u3 + u1*u2*u3**2 ) )

        return u   # shape (T,)

    # ---- full energy --------------------------------------------------
    def value(self, u: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
        """
        Compute the full Energy
        ----------
        u          : (N,) nodal values
        triangles  : (T,3) index tensor
        ----------
        E(u)  : scalar tensor (same dtype/device as inputs)
        """
        #  nodal values per triangle
        u_tri = u[triangles]                  # (T,3)
        u1, u2, u3 = u_tri.unbind(dim=1)      # each (T,)

        # ---- potential part ------------------------------------------
        pot = self._potential_per_triangle(u1, u2, u3).sum() / (4*self.eps)

        # ---- Dirichlet part ------------------------------------------
        # gradients: ∑_i u_i * (a_i,b_i)
        a_b   = self.basis_matrices[:, :, 0:2]                # (T,3,2)
        grads = torch.einsum('t i d, t i -> t d', a_b, u_tri) # (T,2)
        dirich = 0.5 * self.eps * (grads.square().sum(dim=1) * self.areas).sum()
        energy = pot + dirich

        return energy
    
    @staticmethod
    def gradient(energy: torch.tensor, u: torch.tensor):
        return torch.autograd.grad(energy, u, create_graph=True, retain_graph=True)[0]
    
    @staticmethod
    def hessian_torch(u_grad):
        pass

    @staticmethod
    def hessian_efficient(E_grad: torch.tensor, u: torch.tensor, edge_list: torch.tensor):
        """
        Vectorised extraction of Hessian entries H_{ij} on an edge list.

        Returns : (E,) tensor   Hij in the same order as `edges`.
        Cost    : one reverse pass for the gradient (already done) +
                  *one* reverse pass for a block of Hessian rows ⇒ O((B+E)·1)
                where B = #distinct row indices in `edges`.
        """
        # ---- gather distinct row indices we really need -----------------
        row_i = edge_list[:, 0]                                  # (E,)
        unique_i, inv_row = torch.unique(row_i, return_inverse=True)   # (B,), (E,)
        B, N = unique_i.numel(), u.numel()

        # ---- build one‑hot selector L (B × N) ---------------------------
        L = torch.zeros((B, N), dtype=u.dtype, device=u.device)
        L[torch.arange(B, device=u.device), unique_i] = 1.0

        # ---- H_rows =  L @ H     shape (B, N) ---------------------------
        H_rows = torch.autograd.grad(
                    E_grad, u,
                    grad_outputs=L,
                    retain_graph=False,
                    is_grads_batched=True        # key flag!
                )[0]                             # (B, N)

        # ---- assemble dense tensor with zeros elsewhere -----------------
        H = torch.zeros((N, N), dtype=u.dtype, device=u.device)
        H[unique_i] = H_rows                 # fill only needed rows

        idx_i, idx_j = edge_list[:,0], edge_list[:,1]
        H_ij = H_rows[inv_row, idx_j]

        H[idx_i, idx_j] = H_ij
        H[idx_j, idx_i] = H_ij
        
        return H                            # (N,N)