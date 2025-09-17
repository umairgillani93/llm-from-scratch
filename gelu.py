import os 
import sys 
import torch
import torch.nn as nn

class GeLU(nn.Module):
    '''
    Activation function for Large language models
    '''
    def __init__(self):
        super().__init__()

    def forward(self, x):
        '''implements Gelu formula'''
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
            ))

#if __name__ == "__main__":
#    torch.manual_seed(123)
#    x = torch.randn(4,5)
#    gelu = GeLU()
#    print(gelu.forward(x))


