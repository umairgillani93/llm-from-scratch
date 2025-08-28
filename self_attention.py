import os 
import sys 
import torch
import torch.nn as nn 

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()

        # Randomly initialized weights
        self.key_weights = nn.Parameter(torch.rand(d_in, d_out))
        self.query_weights = nn.Parameter(torch.rand(d_in, d_out))
        self.value_weights = nn.Parameter(torch.rand(d_in, d_out))

        self.d_in = d_in
        self.d_out = d_out

    def forward(self, x):

        # finding actual vectors by matrix-multiplication with initialized weights
        print(x.shape)
        print(self.key_weights.shape)
        keys = x @ self.key_weights
        values = x @ self.value_weights
        query = x @ self.query_weights

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

    torch.manual_seed(123)
    x = torch.randn(6, 3)
    sa_v1 = SelfAttention_v1(3, 2)
    print(sa_v1.forward(x).sum(dim = -1))


