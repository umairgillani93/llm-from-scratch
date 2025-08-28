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

torch.manual_seed(123)

# dimentions in and out for query, key and value vectors
d_in = 2
d_out = 3

# inititlize the query, kye and value weights vectors
w_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
w_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
w_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)

# some randome input it may be a word and having some vector-embeddings
# of size 3 * 1
x = torch.randn(3, 1)

    
# now using weights and inputs we'll calculate query, key and value vectors
key_vector= w_key @ x
query_vector= w_query @ x
value_vector= w_value @ x


# the un-scaled attention score calculations
                 # word
            # x(x) -> [0.1, 1.3, -0.2]
            #   |                   \
            # wk -> [0.2, 2.3, 2.2]  wv -> [2.3, 3.1, 4.1]
            #   |                           |
            # wq -> [0.4, 0.2]              |
            #   |                           |
            #   \                          /
            #          context vector [0.2, 0.3, -0.1]


