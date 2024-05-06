import math
import torch
from torch import nn, Tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size:int, hidden_size:int, max_seq_len:int) -> None:
        super(PositionalEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.positional_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, input_seq:str) -> Tensor:
        if input_seq.dtype != torch.long:
            input_seq = input_seq.long()

        position = torch.arange(input_seq.shape[-1], dtype=torch.long).unsqueeze(0)
        w_embedding = self.word_embedding(input_seq)
        p_embedding = self.positional_embedding(position)
        embedding = w_embedding + p_embedding
        embedding = self.relu(embedding)
        return self.dropout(embedding)
    

class SelfAttention(nn.Module):
    def __init__(self, hidden_size:int, head_dim:Tensor):
        super(SelfAttention, self).__init__()
        self.head_dim = torch.tensor(head_dim)
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.value = nn.Linear(hidden_size, head_dim, bias=False)

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, mask:Tensor=None)->Tensor:
        Q, K, V = self.query(Q), self.key(K), self.value(V)
        scores = torch.bmm(Q, K.transpose(-2,-1))/torch.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask==1, float("-inf"))
        
        weights = torch.softmax(scores, dim=-1)
        attention = torch.bmm(weights, V)
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size:int, n_heads:int):
        super(MultiHeadAttention, self).__init__()
        self.head_dim = hidden_size//n_heads
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.heads = nn.ModuleList([SelfAttention(hidden_size, self.head_dim) for _ in range(n_heads)])
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, Q:Tensor, K:Tensor, V:Tensor, mask:Tensor=None)->Tensor:
        concat = torch.cat([head(Q, K, V, mask) for head in self.heads], dim=-1)
        output = self.linear(concat) 
        return output


class FeedForward(nn.Module):
    def __init__(self, hidden_size:int, intermediate_size:int) -> None:
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, intermediate_size)
        self.linear_2 = nn.Linear(intermediate_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
    
    def forward(self, attention_outputs:Tensor)->Tensor:
        x = self.linear_1(attention_outputs)
        x = self.relu(x)
        x = self.linear_2(x)
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size:int, n_heads:int, intermediate_size:int) -> None:
        super(TransformerEncoderLayer, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, n_heads)
        self.feed_forward = FeedForward(hidden_size, intermediate_size)

    def forward(self, embed_input:Tensor, mask:Tensor=None)->Tensor:
        attention = self.attention(embed_input,embed_input, embed_input, mask)
        norm_attn = self.layer_norm_1(attention+embed_input)
        ff_out = self.feed_forward(norm_attn)
        output = self.layer_norm_2(ff_out+norm_attn)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size:int, hidden_size:int, max_seq_len:int,
                 n_heads:int, n_layers:int, intermediate_size:int)->None:
        super(TransformerEncoder, self).__init__()
        self.embedding = PositionalEmbedding(vocab_size, hidden_size, max_seq_len)
        self.layers = nn.ModuleList([TransformerEncoderLayer(hidden_size, n_heads, intermediate_size)
                                     for _ in range(n_layers)])
    
    def forward(self, input_ids:Tensor, mask:Tensor=None)->Tensor:
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class BertForClassification(nn.Module):
    def __init__(self, vocab_size:int, hidden_size:int, max_seq_len:int, n_heads:int, 
                 n_layers:int, intermediate_size:int, n_labels:int) -> None:
        super(BertForClassification, self).__init__()
        self.encoder = TransformerEncoder(vocab_size, hidden_size, max_seq_len, 
                                          n_heads, n_layers,intermediate_size)
        self.dropout = nn.Dropout()
        self.classifier = nn.Linear(hidden_size, n_labels)
    
    def forward(self, input_ids:Tensor, mask:Tensor=None)->Tensor:
        x = self.encoder(input_ids, mask)[:, 0, :]
        x = self.dropout(x)
        return self.classifier(x)


if __name__=="__main__":
    vocab_size = 1000
    hidden_size = 256
    max_seq_len = 200
    n_layers = 8
    n_heads = 8
    ff_hidden_size = 1024
    n_labels=10

    t5_model = BertForClassification(
        vocab_size,
        hidden_size,
        max_seq_len,
        n_heads,
        ff_hidden_size,
        n_layers,
        n_labels
    )
    
    seq = torch.randint(0, 1000, (32, 27), dtype=torch.int16)
    print(f"BertForClassification Output Shape {t5_model(seq).shape}")
    

    