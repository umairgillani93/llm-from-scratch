import os
import sys
import torch
import torch.nn as nn
from casual_attn import CausalAttention

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, ctx_len, dropout, num_heads, qkv_bias = False):
        super().__init__()
        
        # loop throght he number of heads and just stack them up to find a context vector 'z'
        # this 'z' should have dimensions (d_out * num_heads)
        # in out case:
            # x = [1,2,3] -> batch_size = 1, number_tokens per line = 2, and input_dimension = 3
            # d_out = 2
            # number_head = 2
            # 'z' shape = (1, 2, 4) i.e d_out * num_heads
            
        self.heads = nn.ModuleList(
                [CausalAttention(d_in, d_out, ctx_len, dropout, qkv_bias)
                    for _ in range(num_heads)]
                )

    def forward(self, x):
        # This will give us 
        # d_out * num_heads length context vectors
        # z1 + z2
        return torch.cat([head(x) for head in self.heads], dim = -1)



if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.rand(1, 2, 3)
    mha = MultiHeadAttentionWrapper(3,2,3,0.5,2)
    print(mha.forward(x).shape)
