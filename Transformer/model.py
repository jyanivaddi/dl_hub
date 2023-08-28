import math
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy



class LayerNormalization(nn.Module):

    def __init__(self, eps:float=1E-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # alpha, learnable mean
        self.bias = nn.Parameter(torch.zeros(1)) # bias, learnable bias

    def forward(self,x):
        # x: (batch_size, seq_len, hidden_size)
        mean = x.mean(dim=-1, keepdim=True) # (batch, seq_len, 1)
        std = x.std(dim=-1, keepdim=True) # (batch, seq_len, 1)
        return self.alpha*(x-mean)/(std+self.eps) + self.bias



class PosEmbedding:
    def __init__(self, num_pos, d_model):
        self.pe = torch.empty((batch_size, num_pos, d_model),dtype='float16', requires_grad=False)
        self.num_pos = num_pos
        self.d_model = d_model

    def __call__(self):
        """
        Compute the sinusoidal positional encodings according to the 
        following formula:
        PE(pos, 2i) = sin(pos/(10000^(2*i/d_model)))
        PE(pos,2i+1) = cos(pos/10000^(2*i/d_model)))
        """
        _denom = torch.logspace(0, -8, d_model).unsqueeze(0)
        _pos = torch.linspace(0,self.num_pos).unsqueeze(1)
        matrix_loc = _pos*_denom
        pe = torch.empty(d.shape)
        for i in range(self.d_model.shape[-1]//2):
            pe[:,2*i] = torch.sin(matrix_loc[:,2*i])
            pe[:, 2*i+1] = torch.cos(matrix_loc[:, 2*i])
        return pe.unsqueeze(0)


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        """
        d_model: dimension of embedding (512)
        d_ff: expanded dim (1024)
        dropout: dropout percentage
        """
        super().__init__()
        self.linear_1 =  nn.Linear(d_model, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # w2 and b2
    
    def forward(self, x):
        # batch: batch size
        # seq_len: length of the sequence
        # d_model: embedding dimension
        # (batch, seq_len, d_model) -> (batch, seq_len, d_ff) -> (batch, seq_len, d_model)
        x = self.linear_1(x)  # (batch, seq_len, d_ff)
        x = torch.relu(x) # (batch, seq_len, d_ff)
        x = self.dropout(x) # (batch, seq_len, d_ff)
        x = self.linear_x(x) # (batch, seq_len, d_model)
        return x
    

class InputEmbedding(nn.Module):

    def __init__(self, d_model: int, vocab_size: int) -> None:
        """
        vocab_size: total no of words in the input
        d_model: embedding dim (512).
        """
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_model)
        # multiply by sqrt(d_model) to scale the embeddings per the paper
        return self.embedding(x)*math.sqrt(self.d_model) 






            


