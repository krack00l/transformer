import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform_
from typing import List, Optional




class PrepareForMultiHeadAttention(nn.Module):
    def __init__(self,
                 dim_model: int,
                 num_heads: int,
                 dim_k: int,
                 bias: bool):
        
        super().__init__()
        self.linear = nn.Linear(dim_model, num_heads * dim_k, bias=bias)
        self.num_heads = num_heads
        self.dim_k = dim_k
    
    def forward(self, x: torch.Tensor):
        head_shape = x.shape[:-1]
        x = self.linear(x)
        x = x.view(*head_shape, self.num_heads, self.dim_k) #*head_shape раскрываем кортеж

        return x




class MultiHeadAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 dim_model: int,
                 dropout: float = 0.1,
                 bias: bool = True):
        
        super().__init__()
        self.dim_k = dim_model // num_heads # Number of features per head
        self.num_heads = num_heads

        self.query = PrepareForMultiHeadAttention(dim_model=dim_model,
                                                  num_heads=num_heads,
                                                  dim_k=self.dim_k,
                                                  bias=bias)
        self.key = PrepareForMultiHeadAttention(dim_model=dim_model,
                                                  num_heads=num_heads,
                                                  dim_k=self.dim_k,
                                                  bias=bias)
        self.value = PrepareForMultiHeadAttention(dim_model=dim_model,
                                                  num_heads=num_heads,
                                                  dim_k=self.dim_k,
                                                  bias=True)
        self.softmax = nn.Softmax(dim=1) # along the time dimension 
        self.output = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1 / math.sqrt(self.dim_k)

        self.attn = None # for debug
    

    # along the time dimension 
    def get_scores(self,
                   query: torch.Tensor,
                   key: torch.Tensor):
        
        out = torch.einsum('ibhd,jbhd->ijbh', query, key)
        return out

    # mask has shape [seq_len_q, seq_len_k, batch_size]
    def prepare_mask(self,
                     mask: torch.Tensor,
                     query_shape: List[int],
                     key_shape: List[int]):
        
        assert mask.shape[0] == 1 or mask.shape[0] == query_shape[0]
        assert mask.shape[1] == key_shape[0]
        assert mask.shape[2] == 1 or mask.shape[2] == query_shape[1]

        mask = mask.unsqueeze(-1)

        return mask
    

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                mask: Optional[torch.Tensor] = None):
        
        # query , key and value are the tensors that store collection of query, 
        # key and value vectors. They have shape [seq_len, batch_size, d_model] .
        # mask has shape [seq_len, seq_len, batch_size] and mask[i, j, b] 
        # indicates whether for batch b , query at position i has access to key-value at position j.

        seq_len, batch_size, _ = query.shape

        if mask is not None:
            mask = self.prepare_mask(mask, query.shape, key.shape)
        
        query = self.query(query)
        key = self.key(key)
        value = self.value(value)
        # Scale scores
        scores = self.get_scores(query, key)
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn = self.softmax(scores)
        attn = self.dropout(attn)
        x = torch.einsum("ijbh,jbhd->ibhd", attn, value)
        x = x.reshape(seq_len, batch_size, -1)
        out = self.output(x)

        return out