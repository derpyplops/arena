from collections import OrderedDict
import torch as t
import torch.nn as nn
from torch import Tensor
from fancy_einsum import einsum
from einops import rearrange, repeat
import math

from dataclasses import dataclass

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
    device = 'cpu'

# config = TransformerConfig(
#     num_layers = 6,
#     num_heads = 4,
#     vocab_size = 10,
#     hidden_size = 96
# )

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()

        d = d_model
        L = max_len
        D = d / 2

        angles = t.outer(t.arange(L), 1 / 10000 ** (2 * t.arange(D) / D))

        array_2d = t.zeros((L, d))
        array_2d[:, ::2] = t.sin(angles)
        array_2d[:, 1::2] = t.cos(angles)
        self.encoding = array_2d

    def forward(self, x: Tensor) -> Tensor:
        '''
        x: Tensor, shape [batch, seq_len, embedding_dim]
        ''' 
        batch, seq_len, embedding_dim = x.shape
        return self.encoding[:seq_len, :]

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

    def forward(self, x: t.Tensor):
        return self.model(x)

def multihead_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, n_heads: int):
    '''
    Q: shape (b, s1, e)
    K: shape (b, s2, e)
    V: shape (b, s2, e)

    e = nheads * h
    b = batch
    s = seq_len
    h = hidden

    Return: shape (b s e)
    '''

    assert Q.shape[-1] % n_heads == 0
    assert K.shape[-1] % n_heads == 0
    assert V.shape[-1] % n_heads == 0
    assert K.shape[-1] == V.shape[-1]

    Q = rearrange(Q, 'b s (nheads h) -> b nheads s h', nheads=n_heads)
    K = rearrange(K, 'b s (nheads h) -> b nheads s h', nheads=n_heads)
    V = rearrange(V, 'b s (nheads h) -> b nheads s h', nheads=n_heads)

    scaled_dot_prod = einsum('b nheads s1 h, b nheads s2 h -> b nheads s2 s1', K, Q) / math.sqrt(Q.shape[-1])
    mask_filter = t.triu(t.full_like(scaled_dot_prod, -t.inf), 1)
    scaled_dot_prod += mask_filter
    attention_probs = scaled_dot_prod.softmax(dim=-1)
    attention_vals = einsum('b nheads s1 s2, b nheads s2 c -> b nheads s1 c', attention_probs, V)
    attention = rearrange(attention_vals, 'b nheads s c -> b s (nheads c)')
    return attention

class MultiheadMaskedAttention(nn.Module):

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.W_QKV = nn.Linear(hidden_size, hidden_size * 3)
        self.W_O = nn.Linear(hidden_size, hidden_size)
        self.num_heads = num_heads

    def forward(self, x: t.Tensor, mask=None) -> t.Tensor:
        '''
        x: shape (batch, seq, hidden_size)
        Return: shape (batch, seq, hidden_size)
        '''
        Q, K, V = self.W_QKV(x).chunk(3, dim=-1)
        att = multihead_attention(Q, K, V, self.num_heads)
        return self.W_O(att)

class DecoderBlock(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiheadMaskedAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_heads
        )
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.mlp = MultiLayerPerceptron(config.hidden_size, config.hidden_size)
    
    def forward(self, x: t.Tensor):
        h1 = self.layernorm1(self.attention(x) + x)
        h2 = self.layernorm2(self.mlp(h1) + h1)
        return h2

class DecoderTransformer(nn.Module):

    def __init__(self, config: TransformerConfig):
        super().__init__()
        decoders = [DecoderBlock(config) for i in range(config.num_layers)]
        names = ['decoder' + str(i) for i in range(config.num_layers)]
        self.decoderlayer = nn.Sequential(OrderedDict(zip(names, decoders)))
        self.dropout = nn.Dropout(p=config.dropout)
        self.layernorm = nn.LayerNorm(config.hidden_size) # why? come back to this later
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_embedding = PositionalEncoding(config.hidden_size)
        self.last_linear = nn.Linear(config.hidden_size, config.vocab_size)

    def forward(self, tokens):
        embedding = self.embed(tokens) # (seq_len) -> (seq_len, embedding)
        pos_embedding = self.positional_embedding(embedding)
        final_embedding = embedding + pos_embedding
        a = self.dropout(final_embedding)
        b = self.decoderlayer(a)
        c = self.layernorm(b) @ self.embed.weight.T
        return c