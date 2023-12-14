

import torch
from model import model


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Number of trainable parameters: {count_parameters(model)}")


checkpoint = torch.load('models/v3_epoch_65.pth')
print(checkpoint.keys())
print({k: v.size() for k, v in checkpoint.items()})


checkpoint = torch.load('models/v5_epoch_55.pth')
print(checkpoint.keys())
print({k: v.size() for k, v in checkpoint.items()})
