import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    '''Root-Mean-Square Normalization'''
    def __init__(self, features, eps=1e-6):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()

        return self.gamma * x/rms
    

features = 768
rms_norm = RMSNorm(features)

x = torch.randn(10, 20, features)
normalized_x = rms_norm(x)
print(normalized_x)
