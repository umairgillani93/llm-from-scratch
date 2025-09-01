'''
Implementing DummpyChatGPT model
'''
import os 
import sys 
import torch
import tiktoken
import config
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

class DummyGptModel(nn.Module):
    def __init__(self, cfg):
        '''passing configuration file'''
        # inherit functions from base
        super().__init__()
        # input token embedding size
        self.token_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['ctx_len'], cfg['emb_dim'])

        # drop-out ratio
        self.drop_emb = nn.Dropout(cfg['dropout_rate'])
        
        # stack the transformers blocks sequentially
        self.trf_block = nn.Sequential(
                *[DummyTransformerBlock(cfg) for _ in range(cfg['num_layers'])]
                )

        # Adding layer normalization to help with stable training.
        self.final_norm = DummyLayerNorm(cfg['emb_dim'])

        # Final logits: defines the score for each word in vocab against each token
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias = False)

    def forward(self, x):
        '''Forward pass for gpt_model_class.'''
        # Embedding layer takes two inputs:
        # batch_size and sequence length and returns 
        # the embeddings agaisnt each toekn

        batch_size, ctx_len = x.shape
        token_embedding = self.token_emb(x)

        # Define positional embeddings
        # Creates a tensor of 0 - seq_len - 1 and then passes to Embedding layer
        pos_embedding = self.pos_emb(torch.arange(ctx_len, device = x.device))

        x = token_embedding + pos_embedding

        # Add dropout
        x = self.drop_emb(x)
        x = self.trf_block(x)
        x = self.final_norm(x)

        # logits
        logits = self.out_head(x)
        return logits


class DummyTransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()

    def forward(self, x):
        return x

class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5):
        super().__init__()

    def forward(self, x):
        return x
    

if __name__ == '__main__':
    torch.manual_seed(123)
    text_1 = "First row of items"
    text_2 = "Second row of items"

    # initialize a tokenizer
    tokenizer = tiktoken.get_encoding('gpt2')

    # Create a batch of text.
    batch = []
    batch.append(torch.tensor(tokenizer.encode(text_1)))
    batch.append(torch.tensor(tokenizer.encode(text_2)))

    batch = torch.stack(batch, dim = 0)

    #batch = pad_sequence(batch, batch_first = True, padding_value = tokenizer.pad_token_id)
    print(f'batch: {batch}')
    print(f'batch type: {type(batch)}')

    GPT_CONFIG = config.Config
    model = DummyGptModel(GPT_CONFIG)


    # logits
    logits = model.forward(batch)

    print(f'logits: {logits}')
    print(logits.shape)
