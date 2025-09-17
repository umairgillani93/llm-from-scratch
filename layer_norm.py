import os 
import sys 
import torch 
import tiktoken
import torch.nn as nn 

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift= nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim= True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


if __name__ == "__main__":
    torch.manual_seed(123)
    text = 'this is a temporary test text'

    # initialize a tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    batch = torch.tensor(
            tokenizer.encode(text)
            )
    print(batch)
    # Create a batch of text.
    #batch = []
    #batch.append(torch.tensor(tokenizer.encode(text_1)))
    #batch.append(torch.tensor(tokenizer.encode(text_2)))
    #batch = torch.stack(batch, dim = 0)

    ln = LayerNorm(emb_dim = 6)
    out_ln = ln.forward(batch.float())
    mean = out_ln.mean(dim = -1, keepdim = True)
    print(mean)
