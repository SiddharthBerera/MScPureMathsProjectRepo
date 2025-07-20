import torch

def gradient(energy: torch.tensor, u: torch.tensor):
        return torch.autograd.grad(energy, u, create_graph=True, retain_graph=True)[0]