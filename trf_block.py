import os 
import sys 
import torch 
import torch.nn as nn 
from multi_head_attn_2 import MultiHeadAttention
from layer_norm import LayerNorm
from feed_forward_nn import FF 
from config import Config


class TransformerModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.att = MultiHeadAttention(
                d_in = conf['emb_dim'],
                d_out = conf['emb_dim'],
                num_heads = conf['num_heads'],
                ctx_len = conf['ctx_len'],
                dropout = conf['dropout_rate'],
                qkv_bias = conf['qkv_bias'],
                )
        
        self.ff = FF(conf)
        self.norm1 = LayerNorm(conf['emb_dim'])
        self.norm2 = LayerNorm(conf['emb_dim'])
        self.dropout = nn.Dropout(conf['dropout_rate'])


    def forward(self, x):
        '''forward pass for transformers block'''
        # initiazlie the skip-connection
        shortcut = x
        print(f'x shape: {x.shape}')
        x = self.norm1(x)
        x = self.att(x)
        x = self.dropout(x)
        x = x + shortcut # add skip-connection
        
        # assing shortcut the current 'x'
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.dropout(x)
        x = x + shortcut 
        return x

if __name__ == "__main__":
    torch.manual_seed(123)
    x = torch.randn(2, 4, 768)
    c = Config
    print(f'Model initializgin..')
    tf_model = TransformerModel(c)
    print(f'Running forward pass..')
    block = tf_model(x)
    print(block)
    print(f'Tensor shape: {block.shape}')

    
