import torch

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

'''
print(bary_integral(4,0,0))  # ∫ λ_i⁴
print(bary_integral(3,1,0))   # ∫ λ_i³ λ_j
print(bary_integral(2,2,0))  # ∫ λ_i² λ_j²
print(bary_integral(2,1,1))   # ∫ λ_i² λ_j λ_k
print(bary_integral(2,0,0))   # ∫ λ_i²
print(bary_integral(1,1,0))   # ∫ λ_i λ_j
'''