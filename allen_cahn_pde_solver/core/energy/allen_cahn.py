import torch
import math

class AllenCahnEnergy:
    def __init__(self, area: torch.Tensor, basis_matrix: torch.Tensor, eps: float):
        """
        area: PyTorch tensor shape (T,) of each triangles area
        basis_matrix: (T,N,N) PyTorch tensors of basis matrix for each triangle
        eps : interface thickness
        """
        self.areas = area
        self.basis_matrices = basis_matrix
        self.eps = eps
        self.bi4 = 2 * ( math.factorial(4) ) / math.factorial(6)
        self.bi22 = 2 * ( math.factorial(2)*math.factorial(2) ) / math.factorial(6)
        self.bi31 = 2 * ( math.factorial(3) ) / math.factorial(6)
        self.bi211 = 2 * ( math.factorial(2) ) / math.factorial(6)

    '''
    # integral of (phi_1)^a * (phi_2)^b * (phi_3)^b
    def basis_integrals(self, a,b,c, area):
        I = 2 * ( math.factorial(a)*math.factorial(b)*math.factorial(b) ) / math.factorial(a+b+c+2) * area
        return I
    '''

    # polynomial for potential term after integration on a triangle with P1 basis functions
    def potential_poly(self, u1, u2, u3, area):
        u = area * ( self.bi4 * ( u1**4 + u2**4 + u3**4 ) 
        + 2 * self.bi22 * ( (u1*u2)**2 + (u1*u3)**2 + (u2*u3)**2 )
        + 4 * self.bi31 * ( u1*u2**3 + u1**3*u2 + u1*u3**3 + u1**3*u3 + u2*u3**3 + u2**3*u3)
        + 12 * self.bi211 * (u1**2*u2*u3 + u1*u2**2*u3 + u1*u2*u3**2 ) )

        return u

    def value(self, u: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
        u_tris = u[triangles]
        u1s, u2s, u3s = u_tris.unbind(dim=1)

        a_b = self.basis_matrices[:, :, 0:2]
        
        # Potential part: w = 1 - u^2
        # w = 1 - u**2
        pot = (self.potential_poly(u1s, u2s, u3s, self.areas)).sum() / (4 * self.eps)

        # Dirichlet part
        grads = torch.einsum('t i d, t i -> t d', a_b, u_tris)
        dirich = ( 0.5 * self.eps * (grads**2).sum(dim=1) * self.areas ).sum()

        return dirich + pot
