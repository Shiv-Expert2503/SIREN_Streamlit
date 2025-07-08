import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from PIL import Image
import torch
import torch.nn as nn
import math 

# Creating sine activation function 
class sine_af(nn.Module):
    def __init__(self, w0 = 1):
        super().__init__()
        self.w0 = w0


    def forward(self, x):
        return torch.sin((self.w0)*x)
    
def siren_self_weights(layer, is_first_layer = False, w0_initial = 30.0):
    if isinstance(layer, nn.Linear):
        fan_in = layer.in_features
    
        if is_first_layer:
            limit = 1/ fan_in
    
        else:
            limit = (math.sqrt(6/fan_in))/w0_initial
    
        layer.weight.data.uniform_(-limit, limit)
        layer.bias.data.fill_(0)

class Siren(nn.Module):
    def __init__(self, inputs,hidden_features, hidden_layers, output_number, frequency_initial = 30, frequency = 1):
        super(Siren, self).__init__()

        self.lists = []
    
        # First Layer
        first_layer = nn.Linear(inputs, hidden_features)
        siren_self_weights(first_layer, is_first_layer = True, w0_initial = frequency_initial)
        self.lists.append(first_layer)
        # applying AF
        self.lists.append(sine_af(w0 = frequency_initial))
    
        # Hidden Layers
        for hidden_layer in range(hidden_layers):
            
            layer = nn.Linear(hidden_features, hidden_features)
            siren_self_weights(layer, is_first_layer=False, w0_initial = frequency)
            self.lists.append(layer)
            # Applying AF
            self.lists.append(sine_af(w0 = frequency))

        # Final layer
        final_layer = nn.Linear(hidden_features, output_number)
        siren_self_weights(final_layer, is_first_layer = False, w0_initial = frequency)
        self.lists.append(final_layer)

        self.net = nn.Sequential(*self.lists)


    def forward(self, x):
        return self.net(x)