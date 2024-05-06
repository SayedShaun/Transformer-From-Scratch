import torch
from torch import nn, Tensor


class PositionalEmbedding(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_seq_len: int):
        super(PositionalEmbedding, self).__init__()
        self.word_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_seq_len, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()

    def forward(self, input_seq: Tensor) -> Tensor:
        position = torch.arange(input_seq.shape[-1], dtype=torch.long).unsqueeze(0)
        w_embedding = self.word_embedding(input_seq.long())
        p_embedding = self.position_embedding(position)
        embedding = w_embedding + p_embedding
        return self.dropout(self.relu(embedding))


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, head_dim: Tensor):
        super(SelfAttention, self).__init__()
        self.head_dim = torch.tensor(head_dim)
        self.query = nn.Linear(hidden_size, head_dim, bias=False)
        self.key = nn.Linear(hidden_size, head_dim, bias=False)
        self.value = nn.Linear(hidden_size, head_dim, bias=False)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor = None) -> Tensor:
        Q, K, V = self.query(Q), self.key(K), self.value(V)
        scores = torch.bmm(Q, K.transpose(-2, -1)) / torch.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        attention = torch.bmm(weights, V)
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int):
        super(MultiHeadAttention, self).__init__()
        self.head_dim = hidden_size // n_heads
        assert hidden_size % n_heads == 0, "hidden_size must be divisible by n_heads"
        self.heads = nn.ModuleList([SelfAttention(hidden_size, self.head_dim) for _ in range(n_heads)])
        self.linear = nn.Linear(hidden_size, hidden_size)

    def forward(self, Q: Tensor, K: Tensor, V: Tensor, mask: Tensor = None) -> Tensor:
        concat = torch.cat([head(Q, K, V, mask) for head in self.heads], dim=-1)
        output = self.linear(concat)
        return output


class FeedForward(nn.Module):
    def __init__(self, hidden_size: int, ff_hidden_size: int):
        super(FeedForward, self).__init__()
        self.linear_1 = nn.Linear(hidden_size, ff_hidden_size)
        self.linear_2 = nn.Linear(ff_hidden_size, hidden_size)
        self.dropout = nn.Dropout()
        self.gelu = nn.GELU()

    def forward(self, attn_output):
        x = self.linear_1(attn_output)
        x = self.gelu(x)
        x = self.linear_2(x)
        return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, ff_hidden_size: int):
        super(TransformerEncoderBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, n_heads)
        self.feed_forward = FeedForward(hidden_size, ff_hidden_size)

    def forward(self, src_seq: Tensor, src_mask: Tensor = None) -> Tensor:
        # Self Attention
        attn_out = self.attention(src_seq, src_seq, src_seq, src_mask)
        x = self.layer_norm_1(attn_out + src_seq)

        # Feed Forward
        ff_out = self.feed_forward(x)
        output = self.layer_norm_2(ff_out + x)
        return output


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_seq_len: int,
                 n_heads: int, ff_hidden_size: int, n_layers: int) -> None:
        super(TransformerEncoder, self).__init__()
        self.embedding = PositionalEmbedding(vocab_size, hidden_size, max_seq_len)
        self.blocks = nn.ModuleList([TransformerEncoderBlock(hidden_size, n_heads, ff_hidden_size)
                                     for _ in range(n_layers)])

    def forward(self, input_seq: Tensor, src_mask: Tensor = None) -> Tensor:
        embed_seq = self.embedding(input_seq)
        encoder_out = None
        for block in self.blocks:
            encoder_out = block(embed_seq, src_mask)
        return encoder_out


class TransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int, n_heads: int, ff_hidden_size: int) -> None:
        super(TransformerDecoderBlock, self).__init__()
        self.layer_norm_1 = nn.LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, n_heads)
        self.cross_attention = MultiHeadAttention(hidden_size, n_heads)
        self.encoder_block = TransformerEncoderBlock(hidden_size, n_heads, ff_hidden_size)
        self.layer_norm_2 = nn.LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, ff_hidden_size)
        self.layer_norm_3 = nn.LayerNorm(hidden_size)

    def forward(self, trg_seq: Tensor, encoder_out: Tensor, trg_mask: Tensor = None) -> Tensor: #Need to Add Src_Mask
        # Kay, Value will come from Encoder
        key = value = encoder_out
        # Self Attention
        self_attn = self.attention(trg_seq, trg_seq, trg_seq, trg_mask)
        query = self.layer_norm_1(self_attn + trg_seq)
        # Cross Attention
        cross_attn = self.cross_attention(query, key, value)
        cross_attn_out = self.layer_norm_2(cross_attn + query)
        # Feed Forward
        ff_out = self.feed_forward(cross_attn_out)
        output = self.layer_norm_3(ff_out + cross_attn_out)

        return output


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, max_seq_len: int,
                 n_heads: int, ff_hidden_size: int, n_layers: int) -> None:
        super(TransformerDecoder, self).__init__()
        self.embedding = PositionalEmbedding(vocab_size, hidden_size, max_seq_len)
        self.decoder_block = nn.ModuleList([TransformerDecoderBlock(hidden_size, n_heads, ff_hidden_size)
                                            for _ in range(n_layers)])

    def forward(self, input_seq: Tensor, encoder_out: Tensor, target_mask: Tensor = None) -> Tensor:
        x = self.embedding(input_seq)
        transformer_out = None
        for block in self.decoder_block:
            transformer_out = block(x, encoder_out, target_mask)
        return transformer_out


class T5(nn.Module):
    def __init__(self, src_vocab: int, trg_vocab: int, hidden_size: int, max_seq_len: int,
                 n_heads: int, n_layers: int, ff_hidden_size: int) -> None:
        super(T5, self).__init__()
        self.encoder = TransformerEncoder(src_vocab, hidden_size, max_seq_len,
                                          n_heads, n_layers, ff_hidden_size)
        self.decoder = TransformerDecoder(trg_vocab, hidden_size, max_seq_len,
                                          n_heads, n_layers, ff_hidden_size)
        self.linear = nn.Linear(hidden_size, trg_vocab)
        self.dropout = nn.Dropout()

    def forward(self, src_seq: Tensor, trg_seq: Tensor, trg_mask: Tensor, src_mask: Tensor = None) -> Tensor:
        encoder_output = self.encoder(src_seq, src_mask)
        decoder_output = self.decoder(trg_seq, encoder_output, trg_mask)
        output = self.linear(decoder_output)
        return self.dropout(output)


if __name__ == "__main__":
    src_vocab = 1000
    trg_vocab = 1500
    hidden_size = 256
    max_seq_len = 200
    n_layers = 8
    n_heads = 8
    ff_hidden_size = 1024

    t5_model = T5(
        src_vocab,
        trg_vocab,
        hidden_size,
        max_seq_len,
        n_heads,
        ff_hidden_size,
        n_layers
    )

    source_seq = torch.randint(0, 1000, (32, 27), dtype=torch.int16)
    target_seq = torch.randint(0, 1000, (32, 27), dtype=torch.int16)

    trg_mask = torch.tril(torch.ones((target_seq.shape[1], target_seq.shape[1])))
    src_mask = (source_seq != 0).unsqueeze(1)
    print(f"T5 Output Shape {t5_model(source_seq, target_seq, trg_mask, src_mask).shape}")
