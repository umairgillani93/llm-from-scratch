import os 
import sys 
import torch
import torch.nn as nn 

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()

        # Randomly initialized weights
        self.key_weights = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.value_weights = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.query_weights = nn.Linear(d_in, d_out, bias = qkv_bias)

        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x):

        # finding actual vectors by matrix-multiplication with initialized weights
        keys = self.key_weights(x)
        query = self.query_weights(x)
        values = self.value_weights(x)

        # finding attention scores
        attn_scores = query @ keys.T 

        # attn_weights -> Normalizing using softmax
        attn_weights = torch.softmax(
                attn_scores / keys.shape[-1] ** 0.5, dim = -1)

        ctx_vector = attn_weights @ values

        return ctx_vector

if __name__ == '__main__':
    # x -> inputs = [batch_size, input_dimension]
    # weights -> = [input_dimention, output_dimension]

    torch.manual_seed(786)
    x = torch.randn(6, 3)
    sa_v1 = SelfAttention_v1(3, 2)
    print(sa_v1.forward(x))


