import torch

class LayerNorm(torch.nn.Module):
    def __init__(self, shape: int, eps=1e-5):
        super().__init__()

        self.eps = eps
        self.a = torch.nn.Parameter(torch.ones(shape))
        self.b = torch.nn.Parameter(torch.zeros(shape))

    def forward(self, batch):
        mean, var = batch.mean(-1).unsqueeze(-1), batch.var(-1).unsqueeze(-1)
        return (batch - mean) * self.a / torch.sqrt(var + self.eps) + self.b
