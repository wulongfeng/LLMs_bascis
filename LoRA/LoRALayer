import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    def __init__(self, rank, original_layer, alpha):
        super(LoRALayer, self).__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        
        # Freeze the parameters of original layer
        for param in self.original_layer.parameters():
            param.requires_grad = False

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        self.A = nn.Parameter(torch.randn(in_features, rank))
        self.B = nn.Parameter(torch.zeros(rank, out_features))
        
        nn.init.xavier_uniform_(self.A)

    def forward(self, x):
        original_output = self.original_layer(x)

        delta_W = self.alpha * torch.matmul(self.A, self.B) / self.rank
        lora_output = torch.matmul(x, delta_W)

        return original_output + lora_output
    
    def merge_weights(self):
        
        with torch.no_grad():
            delta_W = self.alpha * torch.matmul(self.A, self.B) / self.rank
            self.original_layer.weight.data += delta_W.t()

    