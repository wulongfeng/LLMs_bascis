import torch
import torch.nn as nn

class layerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=False):
        super(layerNorm, self).__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.gamma = None
            self.beta = None
    
    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        x_normalized = (x - mean)/torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x_normalized = self.gamma * x_normalized + self.beta

        return x_normalized

x = torch.tensor([[1,2,3],[4,5,6]], dtype=torch.float32)
layer_norm = layerNorm(normalized_shape=x.size()[1])
output = layer_norm(x)
print(output)