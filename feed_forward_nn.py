import os
import sys 
import torch
import torch.nn as nn
from config import Config
from gelu import GeLU

class FF(nn.Module):
    '''
    Feed forward NN for LLMs
    '''
    def __init__(self, c):

        # INherits from nn.Module
        super().__init__()

        self.layers = nn.Sequential(
                nn.Linear(c['emb_dim'], 4 * c['emb_dim']),
                GeLU(),
                nn.Linear(4 * c['emb_dim'], c['emb_dim'])
                )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    torch.manual_seed(123)
    # generate random batch of samples having dimensions
    # (4, 16, 768)
    x = torch.randn(4, 16, 768)
    ff = FF(Config)
    print(ff.forward(x).shape)
 


