import torch as t
import torch.nn as nn
from torch import Tensor

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
        batch_size, seq_len = x.size()
        return self.encoding[:seq_len, :]

