import os
import sys 
import torch
import torch.nn as nn
from gelu import GeLU

class SkipConn(nn.Module):
    """
    implements the skip-connections class
    """
    def __init__(self, layer_sizes, skip = False):
        # we neend to add 5 layers
        super().__init__()
        self.skip= skip 
        self.layers = nn.ModuleList([
                nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GeLU()),
                nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GeLU()),
                nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GeLU()),
                nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GeLU()),
                #nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GeLU()),
                ])

    def forward(self, x):
        for layer in self.layers:
            output = layer(x)

            # check if shortcut can be applied a
            if self.skip and output.shape == x.shape:
                x = x + output

            else:
                x = output

        return x


if __name__ == '__main__':
    torch.manual_seed(123)
    # x is random n-dimentional vector
    # having shape (4, 16, 786)
    x = torch.randn(4, 3)

    # let's define the layer sizes now
    layer_sizes = 4 * [3] + [1]

    skp_conn = SkipConn(layer_sizes, skip = True)

    print(skp_conn.forward(x))

