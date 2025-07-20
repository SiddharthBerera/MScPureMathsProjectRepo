import torch

def square_mesh(length, paritions):

    n = paritions
    H = length/n
    
    # ------------------------------------------------------------------
    # Vertex list
    # ------------------------------------------------------------------

    grid = torch.linspace(0.0, length, n+1, dtype=torch.float64)
    xx, yy = torch.meshgrid(grid, grid, indexing='ij')
    vertices = torch.stack([xx, yy], dim=-1).reshape(-1,2)
    
    # ------------------------------------------------------------------
    # Edge list  (horizontal, vertical, diagonal)  -> unique pairs
    # ------------------------------------------------------------------
    # Horizontal edges
    j, i = torch.meshgrid(torch.arange(n + 1),   # rows
                          torch.arange(n),   # cols
                          indexing='ij')
    tail_h = j * (n + 1) + i
    head_h = tail_h + 1

    # Vertical edges
    j, i = torch.meshgrid(torch.arange(n),   # rows
                          torch.arange(n + 1),   # cols
                          indexing='ij')
    tail_v = j * (n + 1) + i
    head_v = tail_v + (n + 1)

    # Diagonal edges (along SW–NE split)
    j, i = torch.meshgrid(torch.arange(n),
                          torch.arange(n),
                          indexing='ij')
    tail_d = j * (n + 1) + i
    head_d = tail_d + (n + 2)

    # Stack and sort each pair so min<max
    edges = torch.cat([
        torch.stack([tail_h.flatten(), head_h.flatten()], dim=1),
        torch.stack([tail_v.flatten(), head_v.flatten()], dim=1),
        torch.stack([tail_d.flatten(), head_d.flatten()], dim=1)
    ], dim=0)                                                   # (3m(m+1),2)

    edges = torch.sort(edges, dim=1).values                     # ensure order
    edges = torch.unique(edges, dim=0)       

    # ------------------------------------------------------------------
    # Faces (two per square, SW–NE diagonal)
    # ------------------------------------------------------------------
    v00 = torch.arange(n * n).reshape(n, n)        # (m,m)
    v01 = v00 + 1
    v10 = v00 + (n + 1)
    v11 = v10 + 1

    t1 = torch.stack([v00, v01, v11], dim=-1).reshape(-1, 3)
    t2 = torch.stack([v00, v11, v10], dim=-1).reshape(-1, 3)
    faces = torch.cat([t1, t2], dim=0).long()                      # (T,3)

    # ------------------------------------------------------------------
    # (4) boundary vertex indices
    # ------------------------------------------------------------------
    j, i = torch.meshgrid(torch.arange(n + 1),
                          torch.arange(n + 1),
                          indexing='ij')
    mask   = (j == 0) | (j == n) | (i == 0) | (i == n)
    border = (j * (n + 1) + i)[mask].flatten().long()  # (B,)


    return vertices, edges, faces, border