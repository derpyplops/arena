from dataclasses import dataclass
import torch as t
import torch.nn as nn
import self_attention
from collections import OrderedDict

@dataclass(frozen=True)
class TransformerConfig:
    '''Constants used throughout your decoder-only transformer model.'''

    num_layers: int
    num_heads: int
    vocab_size: int
    hidden_size: int # also embedding dim or d_model
    max_seq_len: int = 5000 
    dropout: float = 0.1
    layer_norm_epsilon: float = 1e-05

config = TransformerConfig(
    num_layers = 6,
    num_heads = 8,
    vocab_size = 10,
    hidden_size = 1
)

class MultiLayerPerceptron(nn.Module):  

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        d_h = d_in * 4
        self.model = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(d_in, d_h)),
            ('GELU', nn.GELU()),
            ('linear2', nn.Linear(d_h, d_in)),   
            ('dropout', nn.Dropout(p=0.1))
        ]))

    def forward(x: t.Tensor):
        return self.model(x)
        
class DecoderBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = self_attention.MultiheadMaskedAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads
        )
        self.layernorm = lambda seq_len: nn.LayerNorm(config.hidden_size)
        self.mlp = MultiLayerPerceptron(config.hidden_size, config.hidden_size)
    
    def forward(x: t.Tensor):
        h1 = self.layernorm(self.attention(x) + x)
        h2 = self.layernorm(self.mlp(h1) + h1)
        return h2

class DecoderTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        decoders = [DecoderBlock(config) for i in range(config.num_layers)]
        names = ['decoder' + str(i) for i in range(config.num_layers)]
        self.decoderlayer = nn.Sequential(OrderedDict(zip(names, decoders)))
        self.dropout = nn.Dropout(p=config.dropout)
        self.layernorm = nn.LayerNorm(config.hidden_size) # why? come back to this later
        self.tokenize = lambda tokens: tokens.unsqueeze(-1) # tokenizer does nothing at the moment

    def forward(self, tokens):
        embedding = tokenize(tokens) # (seq_len) -> (seq_len, embedding)
        pos_embedding = self.pos_emb_layer(tokens)
        final_embedding = embedding + pos_embedding
        a = self.dropout(final_embedding)
        b = self.decoderlayer(a)
        c = self.layernorm(b)
        d = self.unembed(c)
        return d
