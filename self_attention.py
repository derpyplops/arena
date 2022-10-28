import torch as t
import torch.nn as nn
from torch import Tensor
from fancy_einsum import einsum
from einops import rearrange, repeat
import math

def singlehead_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor):
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (b, s, c)
    K: shape (b, s, c)
    V: shape (b, s, c)
    b = batch
    s = seq_len
    c = dims

    Return: shape (b s s)
    '''
    d_k = math.sqrt(Q.shape[-1])
    scaled_dot_prod: Tensor = einsum('b s1 c, b s2 c -> b s1 s2', Q, K) / d_k
    return scaled_dot_prod.softmax(dim=-1) @ V

def masked_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, mask: t.Tensor):
    '''
    Q: shape (b, s, c)
    K: shape (b, s, c)
    V: shape (b, s, c)
    mask: shape (b, s, s)
    b = batch
    s = seq_len
    c = dims

    Return: shape (b s s)
    '''q
    d_k = math.sqrt(Q.shape[-1])
    scaled_dot_prod: Tensor = einsum('b s1 c, b s2 c -> b s1 s2', Q, K) / d_k
    if mask is not None:
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e9)
    return scaled_dot_prod.softmax(dim=-1) @ V

def test_masked_attention():
    Q = t.randn(2, 3, 4)
    K = t.randn(2, 3, 4)
    V = t.randn(2, 3, 4)
    mask = t.zeros(2, 3, 3)
    assert masked_attention(Q, K, V, mask).shape == (2, 3, 3)
    print(mask)
    print(masked_attention(Q, K, V, mask))
    

def test_attention():
    Q = t.randn(2, 3, 4)
    K = t.randn(2, 3, 4)
    V = t.randn(2, 3, 4)
    assert singlehead_attention(Q, K, V).shape == (2, 3, 3)


def multihead_attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor, n_heads: int, mask=None):
    '''
    Q: shape (b, s1, nheads * c)
    K: shape (b, s2, nheads * c)
    V: shape (b, s2, nheads * c)
    mask: shape (s1, s2) OR (b, s1, s2)

    b = batch
    s = seq_len
    c = dims

    Return: shape (b s nheads * c)
    '''

    assert Q.shape[-1] // n_heads == 0
    assert K.shape[-1] // n_heads == 0
    assert V.shape[-1] // n_heads == 0
    assert K.shape[-1] == V.shape[-1]

    Q = rearrange(Q, 'b s (nheads c) -> b nheads s c', nheads=n_heads)
    K = rearrange(K, 'b s (nheads c) -> b nheads s c', nheads=n_heads)
    V = rearrange(V, 'b s (nheads c) -> b nheads s c', nheads=n_heads)

    scaled_dot_prod = einsum('b nheads s1 c, b nheads s2 c -> b nheads s1 s2', K, Q) / math.sqrt(Q.shape[-1])
    if mask is not None:
        if mask.dim() == 2:
            mask = repeat(mask, 's1 s2 -> b s1 s2', b=Q.shape[0])
        else:
            mask = mask.unsqueeze(1)
        scaled_dot_prod = scaled_dot_prod.masked_fill(mask == 0, -1e9)
    attention_probs = scaled_dot_prod.softmax(dim=-1)
    attention_vals = einsum('b nheads s1 s2, b nheads s2 c -> b nheads s1 c', attention_probs, V)
    
    return rearrange(attention_vals, 'b nheads s c -> b s (nheads c)')


def test_multihead_attention():
    Q = t.randn(2, 3, 4)
    K = t.randn(2, 3, 4)
    V = t.randn(2, 3, 4)
    assert multihead_attention(Q, K, V, 2).shape == (2, 3, 4)

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
        return self.W_O(multihead_attention(Q, K, V, self.num_heads, mask))



Q = t.arange(2 * 7 * 3).reshape(2, 7, 3).type(t.float32)
K = Q * 0.5
V = Q * 0.8
print(singlehead_attention(Q,K,V))



