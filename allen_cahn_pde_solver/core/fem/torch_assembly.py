import torch

# Assemble mass and stiffness matrices entirely in PyTorch
def assemble_mass_and_stiffness(points: torch.Tensor,
                                triangles: torch.LongTensor):
    """
    points:   (N,2) float tensor of vertex coords
    triangles:(T,3) long tensor of vertex indices
    returns:
      M: (N,N) sparse tensor
      K: (N,N) sparse tensor
    """
    # Ensure we work in the same dtype as points
    dtype = points.dtype
    device = points.device

    # 1) Extract triangle vertex coordinates: (T,3,2)
    pts_tri = points[triangles]

    # 2) Compute twice-area via cross of edges (vectorized)
    v01 = pts_tri[:,1] - pts_tri[:,0]   # (T,2)
    v02 = pts_tri[:,2] - pts_tri[:,0]
    cross = v01[:,0]*v02[:,1] - v01[:,1]*v02[:,0]  # (T,)
    area = 0.5 * torch.abs(cross)                   # (T,)

    # 3) Local mass entries: Mloc_diag=A/6, Mloc_off=A/12
    Tn = triangles.shape[0]
    # build a (Tn,3,3) tensor of area/12
    Mloc = torch.ones((Tn,3,3), dtype=dtype, device=device) * (area.unsqueeze(-1).unsqueeze(-1)/12)
    Mloc[:, torch.arange(3), torch.arange(3)] = area / 6

    # 4) Local stiffness: compute barycentric grads
    b = torch.empty((Tn,3,2), dtype=dtype, device=device)
    # b_i = [y_j - y_k, x_k - x_j]
    b[:,0,0] = pts_tri[:,1,1] - pts_tri[:,2,1]  # y1 - y2
    b[:,1,0] = pts_tri[:,2,1] - pts_tri[:,0,1]
    b[:,2,0] = pts_tri[:,0,1] - pts_tri[:,1,1]
    b[:,0,1] = pts_tri[:,2,0] - pts_tri[:,1,0]  # x2 - x1
    b[:,1,1] = pts_tri[:,0,0] - pts_tri[:,2,0]
    b[:,2,1] = pts_tri[:,1,0] - pts_tri[:,0,0]
    grads = b / (2 * area.unsqueeze(-1).unsqueeze(-1))  # (T,3,2)

    #    Kloc_ab = (grads_a · grads_b) * area
    Kloc = torch.einsum('t i d, t j d -> t i j', grads, grads) * area.unsqueeze(-1).unsqueeze(-1)

    # 5) Flatten into COO entries
    I = triangles.unsqueeze(-1).expand(-1,3,3).reshape(-1)
    J = triangles.unsqueeze(-1).expand(-1,3,3).permute(0,2,1).reshape(-1)
    M_vals = Mloc.reshape(-1)
    K_vals = Kloc.reshape(-1)

    # 6) Build sparse
    N = points.shape[0]
    M = torch.sparse_coo_tensor(torch.vstack([I,J]), M_vals, (N,N), dtype=dtype, device=device)
    K = torch.sparse_coo_tensor(torch.vstack([I,J]), K_vals, (N,N), dtype=dtype, device=device)

    return M.coalesce(), K.coalesce()