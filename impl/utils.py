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