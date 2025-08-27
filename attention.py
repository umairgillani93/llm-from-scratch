import os 
import sys
import torch

# random inputs # REMEMBER: these should be vectors not scaler values
inputs = torch.randn(6,4)

# matrix multiplications (taking dot product for computing attentions scores)
att_scores = inputs @ inputs.T

# applying softmax in dim = -1 (along the columns for sum = 1 along all rows)
att_weights= torch.softmax(att_scores, dim = -1)

# computing context vectors by taking dot product of input vectors with attention_weights
all_context_vectors = inputs.T @ att_weights

print(all_context_vectors)

