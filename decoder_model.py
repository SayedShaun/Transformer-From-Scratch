# This is the implementation of GPT model

import math
import torch
from torch import nn, Tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size:int, hidden_sie:int, max_seq_len:int):
        super(PositionalEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_sie)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_sie)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout()
    
    def forward(self, input_seq):
        position = torch.arange(input_seq.shape[-1], dtype=torch.long).unsqueeze(0)
        w_embedding = self.word_embedding(input_seq.long())
        p_embedding = self.position_embedding(position)
        embedding = w_embedding + p_embedding
        return self.dropout(self.gelu(embedding))
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_size:int, head_dim:int):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.value = nn.Linear(hidden_size, head_dim, bias=False)

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, mask:Tensor=None)->Tensor:
        Q, K, V = self.query(Q), self.query(K), self.value(V)
        scores = torch.bmm(Q, K.transpose(1, 2))/math.sqrt(K.shape[-1])
        if mask is not None:
            scores = scores.masked_fill(mask==0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attention = torch.bmm(weights, V)
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size:int, n_heads:int):
        super(MultiHeadAttention, self).__init__()
        self.head_dim = hidden_size//n_heads
        assert(self.head_dim * n_heads==hidden_size),"head_dim * n_heads != embed_size"
        self.heads = nn.ModuleList(
            [
                SelfAttention(hidden_size, self.head_dim) 
                for _ in range(n_heads)
            ]
        )
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, mask:Tensor=None)->Tensor:
        concat = torch.cat([head(Q, K, V, mask) for head in self.heads], dim=-1)
        output = self.linear(concat) 
        return output
    

class FeedForward(nn.Module):
    def __init__(self, hidden_size:int, ff_hidden_size:int):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, ff_hidden_size)
        self.linear_2 = nn.Linear(ff_hidden_size, hidden_size)
        self.dropout = nn.Dropout()
        self.gelu = nn.GELU()

    def forward(self, attention_output):
        x = self.linear_1(attention_output)
        x = self.gelu(x)
        x = self.linear_2(x)
        return self.dropout(x)
    

class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_size:int, n_heads:int, ff_hidden_size:int):
        super(TransformerDecoderBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, n_heads)
        self.feed_forward = FeedForward(hidden_size, ff_hidden_size)

    def forward(self, embedded_seq, mask:Tensor=None):
        x = self.layer_norm_1(embedded_seq)
        x = x + self.attention(embedded_seq,embedded_seq,embedded_seq, mask)
        x = self.layer_norm_2(x)
        x = x + self.feed_forward(x)
        return x
    

class TransformerDecoder(nn.Module):
    def __init__(
            self, 
            vocab_size:int, 
            hidden_size:int, 
            max_seq_len:int, 
            n_heads:int, 
            ff_hidden_size:int, 
            n_layers:int
        ):
        super(TransformerDecoder, self).__init__()
        self.embedding = PositionalEmbedding(vocab_size, hidden_size, max_seq_len)
        self.blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(
                    hidden_size, 
                    n_heads, 
                    ff_hidden_size
                )
                for _ in range(n_layers)
            ]
        )

    def forward(self, input_seq, mask:Tensor=None):
        x = self.embedding(input_seq)
        for block in self.blocks:
            x = block(x, mask)
        return x
    

class GPT(nn.Module):
    def __init__(
            self, 
            vocab_size:int, 
            hidden_size:int, 
            max_seq_len:int, 
            n_heads:int, 
            ff_hidden_size:int, 
            n_layers:int,
        ) -> None:
        super(GPT, self).__init__()
        self.decoder = TransformerDecoder(
            vocab_size,
            hidden_size,
            max_seq_len,
            n_heads,
            ff_hidden_size,
            n_layers
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout()
    
    def forward(self, input_seq, mask:Tensor=None):
        x = self.decoder(input_seq, mask)
        output = self.dropout(self.linear(x))
        return output

        
if __name__=="__main__":
    text_generate = GPT(
        vocab_size=1000,
        hidden_size=256,
        max_seq_len=100,
        n_layers=8,
        n_heads=8,
        ff_hidden_size=1024, 
    )
    seq = torch.randint(0, 1000, (32, 27), dtype=torch.int16)
    print(seq)
    mask = torch.tril(torch.ones((seq.shape[1], seq.shape[1])))
    print(f"GPT Output Shape {text_generate(seq, mask).shape}")
