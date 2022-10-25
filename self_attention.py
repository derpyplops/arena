import torch as t
import torch.nn as nn
from torch import Tensor
from fancy_einsum import einsum 

def attention(Q: t.Tensor, K: t.Tensor, V: t.Tensor): 
    '''
    Should return the results of self-attention (see the "Self-Attention in Detail" section of the Illustrated Transformer).

    With this function, you can ignore masking.

    Q: shape (b, s, c)
    K: shape (b, s, c)
    V: shape (b, s, c)
    b = batch
    s = seq_len
    e = embedding (hxc = e)
    h = heads

    Return: shape (FILL THIS IN!)
    '''
    d_k = Q.shape[-1]
    scaled_dot_prod: Tensor = einsum('b s1 c, b s2 c -> b s1 s2', Q, K) / d_k
    return scaled_dot_prod.softmax(dim=-1)

