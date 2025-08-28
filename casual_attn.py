# Implementing casual-attention mechanism for LLMs 
# That considers predicting the next-token on the basis of ONLY previous tokens
# This allows us cancelling out future text to make model
# Understant the language-structure better

import os 
import sys 
import torch
import torch.nn as nn

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, ctx_len, dropout, qkv_bias = False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.key_weights = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.value_weights= nn.Linear(d_in, d_out, bias = qkv_bias)
        self.query_weights= nn.Linear(d_in, d_out, bias = qkv_bias)

        # adding dropout factor to reduce noise and over-fitting
        self.dropout = nn.Dropout(dropout)

        # adding zeros above the diagonal in matrix
        # for triming the future text
        self.register_buffer(
                'mask',
                torch.triu(torch.ones(ctx_len, ctx_len),
                diagonal = 1)
                )


    def forward(self, x):

        # extract values from input -> 3D vector having 
        # batch_size, number_of_tokens each line and input_dimension: (no. of rows)
        batch_size, num_tokens, d_in = x.shape

        keys = self.key_weights(x)
        values = self.value_weights(x)
        query = self.query_weights(x)

        # Finding attn scores
        attn_scores = query @ keys.transpose(1,2)

        # Fill masks
        attn_scores.masked_fill_(
                self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
                )

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)

        # adding the dropout finally
        attn_weights = self.dropout(attn_weights)

        # Final context-vector 'z'
        ctx_vector = attn_weights @ values
        return ctx_vector

if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.rand(1, 2, 3)
    ca = CausalAttention(3,2,3,0.5)
    print(ca.forward(x))
            


        




