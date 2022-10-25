import torch as t
import torch.nn as nn
from torch import Tensor
from fancy_einsum import einsum
import math

def attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor): 
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
    return scaled_dot_prod.softmax(dim=-1)

def test_attention():
    Q = t.randn(2, 3, 4)
    K = t.randn(2, 3, 4)
    V = t.randn(2, 3, 4)
    assert attention(Q, K, V).shape == (2, 3, 3)

test_attention()

