# Multi-head attention implementation 
# To compute the number of heads in parallel 
# Instead of using doing for-loop processing one-by-one

import os 
import sys 
import torch
import torch.nn as nn 

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, ctx_len, dropout, num_heads, qkv_bias = False):
        # we need to reduce the projection dimension
        # to match the output dimension
        super().__init__()
        assert d_out % num_heads == 0 # m
        self.d_out = d_out
        self.d_in = d_in
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.key_weights = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.query_weights= nn.Linear(d_in, d_out, bias = qkv_bias)
        self.value_weights = nn.Linear(d_in, d_out, bias = qkv_bias)

        # Final linear layer projection size
        # We're basically using Linear layer to combine the heads of the output
        self.out_proj = nn.Linear(d_out, d_out)

        # Dropout for reducing noise / over-fitting
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
                'mask', 
                torch.triu(torch.ones(ctx_len,ctx_len), diagonal = 1)
                )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        # Define keys, query and value vectors
        keys = self.key_weights(x)
        query = self.query_weights(x)
        values= self.value_weights(x)
        
        # Re-shape the key, query and values tensors 
        # For multiple attentions head to understand different aspects of sentences.
        # Like, grammer, work-positionaning tone etc.

        # from dimension -> (batch_size, num_tokens, d_in)
        # to dimension -> (batch_size, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        query = query.view(batch_size, num_tokens, self.num_heads, self.head_dim)


        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)
        query = query.transpose(1, 2)


        # compute dot products for each head
        attn_scores = query @ keys.transpose(2, 3)
        
        # mask truncated output to the number of tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        
        # attentions weights
        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim = -1)

        # add drop-out factor
        attn_weights = self.dropout(attn_weights)

        # define context vector
        ctx_vector = (attn_weights @ values).transpose(1, 2)
        
        # (combine heads self.num_heads = self.num_head * self.head_dim)
        ctx_vector = ctx_vector.contiguous().view(batch_size, num_tokens, self.d_out)
        ctx_vector = self.out_proj(ctx_vector)

        return ctx_vector





if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.rand(1, 2, 3)
    print(x)
    mha = MultiHeadAttention(3,2,3,0.5,2)
    print(mha.forward(x).shape)


