# This is implemetetion Transformer 
# ссылка на стать и на реализацию в документации

import torch
import torch.nn as nn
from PositionalEncoding import PositionalEncoding
from MultiHeadAttention import MultiHeadAttention
import math
from feed_forward import FeedForward


#в фид форвард блоке 2 лин слоя используются


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
         dim_model: int,
         num_heads: int,
         dropout: float):
        

        super().__init__()
        self.size = dim_model
        self.multihead_attn = MultiHeadAttention(num_heads=num_heads,
                                                 dim_model=dim_model,
                                                 dropout=dropout)
        self.feed_forward = FeedForward(d_model=dim_model,
                                        d_ff=dim_model*2,
                                        dropout=0.1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([dim_model])
        self.layer_norm_ff = nn.LayerNorm([dim_model])



    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor):
        
        # Multi-Head Attention
        norm_x = self.layer_norm(x)
        self_attn = self.multihead_attn(query=norm_x,
                                        key=norm_x,
                                        value=norm_x,
                                        mask=mask)
        x = x + self.dropout(self_attn)
        # FeedForward
        norm_ff = self.layer_norm_ff(x)
        ff = self.feed_forward(norm_ff)
        x = x + self.dropout(ff)

        return x


class TransformerEncoder(nn.Module):
    def __init__(self,
                 dim_model: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float):
        
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(TransformerEncoderLayer(dim_model=dim_model,
                                                       num_heads=num_heads,
                                                       dropout=dropout))
        self.norm = nn.LayerNorm([dim_model])



    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor):
        
        for layer in self.layers:
            x = layer(x=x, mask=mask)
        out = self.norm(x)

        return out




class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 dim_model: int,
                 num_heads: int,
                 dropout: float):
        
        super().__init__()
        self.size = dim_model
        # Layers
        self.multihead_attn_1 = MultiHeadAttention(num_heads=num_heads,
                                                   dim_model=dim_model,
                                                   dropout=dropout)
        self.multihead_attn_2 = MultiHeadAttention(num_heads=num_heads,
                                                   dim_model=dim_model,
                                                   dropout=dropout)
        self.feed_forward = FeedForward(d_model=dim_model,
                                        d_ff=dim_model*2,
                                        dropout=0.1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm_1 = nn.LayerNorm([dim_model])
        self.layer_norm_2 = nn.LayerNorm([dim_model])
        self.layer_norm_ff = nn.LayerNorm([dim_model])



    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                src: torch.Tensor = None,
                src_mask: torch.Tensor = None):
        
        # Multi-Head Attention 1
        norm_x = self.layer_norm_1(x)
        self_attn = self.multihead_attn_1(query=norm_x,
                                          key=norm_x,
                                          value=norm_x,
                                          mask=mask)
        x = x + self.dropout(self_attn)

        # Multi-Head Attention 2
        norm_x = self.layer_norm_2(x)
        self_attn = self.multihead_attn_2(query=norm_x,
                                          key=norm_x,
                                          value=norm_x,
                                          mask=mask)
        x = x + self.dropout(self_attn)

        # FeedForward
        norm_ff = self.layer_norm_ff(x)
        ff = self.feed_forward(norm_ff)
        x = x + self.dropout(ff)

        return x


    def get_mask(self, sz: int):
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



class TransformerDecoder(nn.Module):
    def __init__(self,
                 dim_model: int,
                 num_heads: int,
                 num_layers: int,
                 dropout: float):
        
        super().__init__()
        self.layers = []
        for i in range(num_layers):
            self.layers.append(TransformerDecoderLayer(dim_model=dim_model,
                                                       num_heads=num_heads,
                                                       dropout=dropout))
        self.norm = nn.LayerNorm([dim_model])


    def forward(self,
                x: torch.Tensor,
                memory: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        
        for layer in self.layers:
            x = layer(x=x,
                      mask=tgt_mask,
                      src=memory,
                      src_mask=src_mask)
        out = self.norm(x)
        return out



class Transformer(nn.Module):
    def __init__(self,
                 encoder: TransformerEncoder,
                 decoder: TransformerDecoder,
                 dim_model: int,
                 n_tokens_encoder: int,
                 n_tokens_decoder: int,
                 dropout: float):
        
        super().__init__()
        self.size = dim_model
        # Layers
        self.encoder = encoder
        self.decoder = decoder
        # Src embedding + positional encoding
        self.src_embed = nn.Embedding(num_embeddings=n_tokens_encoder,
                                      embedding_dim=dim_model)
        self.src_pe = PositionalEncoding(dim_model=dim_model,
                                         dropout=dropout)
        # tgt embedding + positional encoding
        self.tgt_embed = nn.Embedding(num_embeddings=n_tokens_decoder,
                                      embedding_dim=dim_model)
        self.src_pe = PositionalEncoding(dim_model=dim_model,
                                         dropout=dropout)
        
        # Initialize parametrs
        for parametr in self.parameters():
            if parametr.dim() > 1:
                nn.init.xavier_uniform_(parametr)
        
    
    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor,
                tgt_mask: torch.Tensor):
        
        enc = self.encode(src, src_mask)
        out = self.decode(enc, src_mask, tgt,tgt_mask)
        return out
    

    def encode(self,
               src: torch.Tensor,
               src_mask: torch.Tensor):
        
        embed = self.src_embed(src) * math.sqrt(self.size)
        pe = self.src_pe(src)
        res_embed = embed + pe
        return self.encoder(x=res_embed, 
                            src_mask=src_mask)
    

    def decode(self,
               memory: torch.Tensor,
               src_mask: torch.Tensor,
               tgt: torch.Tensor,
               tgt_mask: torch.Tensor):
        
        embed = self.tgt_embed(tgt) * math.sqrt(self.size)
        pe = self.src_pe(tgt)
        res_embed = embed + pe
        return self.decoder(x=res_embed,
                            memory=memory,
                            src_mask=src_mask,
                            tgt_mask=tgt_mask)

if (__name__ == "__main__"):
    encoder = TransformerEncoder(dim_model=100,
                                 num_heads=2,
                                 num_layers=2,
                                 dropout=0.1)
    decoder = TransformerDecoder(dim_model=100,
                                 num_heads=2,
                                 num_layers=2,
                                 dropout=0.1)
    model = Transformer(encoder=encoder,
                    decoder=decoder,
                    dim_model=100,
                    n_tokens_encoder=10012,
                    n_tokens_decoder=12332,
                    dropout=0.1)
    print("all wark succesful")
