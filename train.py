from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, utils
from einops import repeat
from dataclasses import dataclass
import torch as t


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

config = TransformerConfig(
    num_layers = 6,
    num_heads = 8,
    vocab_size = 10,
    hidden_size = 1
)

class TestDataLoader(Dataset):
    """A toy dataset to train a model to reverse
     a random sequence of tokens."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seq_len = 25
        self.total_size = 1000
        self.text = t.rand((self.seq_len,
                                config.hidden_size)).to(config.device).repeat(self.total_size,1,1)
        # self.labels = t.rand((self.seq_len,
        #                         config.hidden_size)).to(config.device).repeat(self.total_size,1,1)
        
    def __len__(self):
        return self.total_size

    def __getitem__(self, idx):
        label = self.text[idx,1:]
        text = self.text[idx,:-1]
        sample = {'text': text, 'label': label}
        return sample

# torch.utils.data.random_split(dataset, lengths, generator=<torch._C.Generator object>)
dl = TestDataLoader(config)


