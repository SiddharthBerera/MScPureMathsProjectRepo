import torch

def hessian_sparse(E_grad: torch.tensor, u: torch.tensor, edge_list: torch.tensor):
        """
        Vectorised extraction of Hessian entries H_{ij} on an edge list.

        Returns : Hessian 2d tensor  - sparse
        Cost    : one reverse pass for the gradient (already done) +
                  *one* reverse pass for a block of Hessian rows ⇒ O((B+E)·1)
                where B = #distinct row indices in `edges`.
        """
        # ---- gather distinct row indices we really need -----------------
        rows = torch.unique(edge_list)   # (B,), (E,)
        B, N = rows.numel(), u.numel()

        # ---- build one‑hot selector L (B × N) ---------------------------
        L = torch.zeros((B, N), dtype=u.dtype, device=u.device)
        L[torch.arange(B, device=u.device), rows] = 1.0

        # ---- H_rows =  L @ H     shape (B, N) ---------------------------
        H_rows = torch.autograd.grad(
                    E_grad, u,
                    grad_outputs=L,
                    retain_graph=True,
                    is_grads_batched=True        # key flag!
                )[0]                             # (B, N)

        # ---- assemble dense tensor with zeros elsewhere -----------------
        H = torch.zeros((N, N), dtype=u.dtype, device=u.device)
        H[rows] = H_rows                 # fill only needed rows

        row_map = torch.full((N,), -1, dtype=torch.long, device=u.device)
        row_map[rows] = torch.arange(B, device=u.device)

        idx_i, idx_j = edge_list[:,0].contiguous(), edge_list[:,1].contiguous()
        
        H_ij = H_rows[row_map[idx_i], idx_j]

        H[idx_i, idx_j] = H_ij
        H[idx_j, idx_i] = H_ij
        
        return H                            # (N,N)