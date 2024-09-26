import torch
import torch.nn
import torch_flops
from torch_flops import TorchFLOPsByFX


class gen(torch.nn.Module):
    def __init__(self, len=10, omega=0.5):
        super(gen, self).__init__()
        