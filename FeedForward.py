import torch
from torch import nn as nn



class FeedForward(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 d_ff: int,
                 dropout: float = 0.1,
                 activation=nn.ReLU(),
                 is_gated: bool = False,
                 bias1: bool = True,
                 bias2: bool = True,
                 bias_gate: bool = True):
        super().__init__()
        # Layer one parameterized by weight W_1 and bias b_1
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        # Layer one parameterized by weight W_1 and bias b_1
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        # Hidden layer dropout
        self.dropout = nn.Dropout(dropout)
        # Activation function f
        self.activation = activation
        # Whether there is a gate
        self.is_gated = is_gated
        if is_gated:
            # If there is a gate the linear layer to transform inputs to
            # be multiplied by the gate, parameterized by weight V and bias c
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        # f(x W_1 + b_1)
        g = self.activation(self.layer1(x))
        # If gated, f(x W_1 + b_1) \otimes (x V + b) 
        if self.is_gated:
            x = g * self.linear_v(x)
        # Otherwise
        else:
            x = g
        # Apply dropout
        x = self.dropout(x)
        # (f(x W_1 + b_1) \otimes (x V + b)) W_2 + b_2 or f(x W_1 + b_1) W_2 + b_2
        # depending on whether it is gated
        return self.layer2(x)