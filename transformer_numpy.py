import numpy as np


class Embedding:
    def __init__(self, num_embeddings:int, embedding_dim:int)->None:
        self.weights = np.random.randn(num_embeddings, embedding_dim)
    
    def __call__(self, input_seq:int)->np.array:
        return self.weights[input_seq]
    

class Linear:
    def __init__(self, in_features:int, out_features:int, bias:bool=True)->None:
        self.is_bias = bias
        self.weights = np.random.randn(out_features, in_features)
        self.bias = np.random.randn(out_features)
        
    def __call__(self, input_seq:np.array)->np.array:
        y = np.matmul(input_seq, self.weights.T)
        if self.is_bias==True:
            y = y + self.bias
        return y
    

class PositionalEmbedding:
    def __init__(self, vocab_size:int, hidden_size:int, max_seq_len:int)->None:
        self.word_embedding = Embedding(vocab_size, hidden_size)
        self.position_embedding = Embedding(max_seq_len, hidden_size)
        self.dropout = Dropout()
        self.relu = Relu()

    def __call__(self, input_seq:np.array)->np.array:
        positions = np.arange(input_seq.shape[-1]).reshape(1,-1)
        w_embedding = self.word_embedding(input_seq)
        p_embedding = self.position_embedding(positions)
        embedding = w_embedding+p_embedding
        return self.dropout(self.relu(embedding)) 
    

class Dropout:
    def __init__(self, dropout_rate:float=0.5)->None:
        self.dropout_rate = dropout_rate

    def __call__(self, input:np.array, training:bool=True)->np.array:
        if not training:
            return input
        mask = np.random.rand(*input.shape) >= self.dropout_rate
        output = (input*mask)/(1-self.dropout_rate)
        return output
    

class Relu:
    def __call__(self, x:np.array)->np.array:
        return np.where(x<0, 0.0, x)


class Softmax:
    def __call__(self, x:np.array)->np.array:
        numerator = np.exp(x)
        denominator = np.sum(np.exp(x))
        return numerator/denominator


class SelfAttention:
    def __init__(self, hidden_size:int, head_dim:int)->None:
        self.head_dim = head_dim
        self.query = Linear(hidden_size, head_dim, bias=False)
        self.key = Linear(hidden_size, head_dim, bias=False)
        self.value = Linear(hidden_size, head_dim, bias=False)
        self.softmax = Softmax()

    def __call__(self, Q:np.array, K:np.array, V:np.array, mask=None)->np.array:
        Q, K, V = self.query(Q), self.key(K), self.value(V)
        scores = np.matmul(Q, K.transpose(0, 2, 1))/np.sqrt(self.head_dim)
        if mask is not None:
               scores = np.ma.masked_fill(scores, mask==0, float("-inf"))
        weights = self.softmax(scores)
        attention = np.matmul(weights, V)
        return attention
    

class MultiHeadAttention:
    def __init__(self, hidden_size, n_heads)->None:
        self.head_dim = hidden_size//n_heads
        assert(self.head_dim*n_heads==hidden_size)
        self.heads = [SelfAttention(hidden_size, self.head_dim) for _ in range(n_heads)]
        self.linear = Linear(hidden_size, hidden_size)
        self.dropout = Dropout()

    def __call__(self, Q:np.array, K:np.array, V:np.array, mask=None)->np.array:
        concat = np.concatenate([head(Q,K,V,mask) for head in self.heads], axis=-1)
        output = self.linear(concat)
        return output


class FeedForward:
    def __init__(self, hidden_size, ff_hidden)->None:
        self.feed_forward_1 = Linear(hidden_size, ff_hidden)
        self.feed_forward_2 = Linear(ff_hidden, hidden_size)
        self.relu = Relu()
        self.dropout = Dropout()

    def __call__(self, seq_embed: np.array) -> np.array:
        x = self.feed_forward_1(seq_embed)
        x = self.relu(x)
        x = self.feed_forward_2(x)
        x = self.relu(x)
        return x
    

class LayerNorm:
    def __init__(self, hidden_size:int, epsilon:float=1e-05)->None:
        self.epsilon = epsilon
        self.gamma = np.ones(hidden_size)
        self.beta = np.zeros(hidden_size)
    
    def __call__(self, x:np.array)->np.array:
        mean = np.mean(x)
        var = np.var(x)
        normalized_x = (x-mean) / np.sqrt(var + self.epsilon)
        output = normalized_x * self.gamma + self.beta
        return output
    

class TransformerEncoderBlock:
    def __init__(self, hidden_size:int, n_heads:int, ff_hidden:int)->None:
        self.layer_norm_1 = LayerNorm(hidden_size)
        self.attention = MultiHeadAttention(hidden_size, n_heads)
        self.layer_norm_2 = LayerNorm(hidden_size)
        self.feed_forward = FeedForward(hidden_size, ff_hidden)

    def __call__(self, source_seq:np.array, mask=None)->np.array:
        x = self.layer_norm_1(source_seq)
        x = x + self.attention(x, x, x, mask)
        x = self.layer_norm_2(x)
        x = x + self.feed_forward(x)
        return x
    

class TransformerEncoder:
    def __init__(self, vocab_size:int, max_seq_len:int, hidden_size:int, n_heads:int, ff_hidden:int)->None:
        self.embedding = PositionalEmbedding(vocab_size, hidden_size, max_seq_len)
        self.layers = [
            TransformerEncoderBlock(hidden_size, n_heads, ff_hidden) for _ in range(n_heads)
        ]

    def __call__(self, input_seq, mask=None):
        embedded_seq = self.embedding(input_seq)
        encoder_out = None
        for layer in self.layers:
            encoder_out=layer(embedded_seq, mask)
        return encoder_out
    

class TransformerDecoderBlock:
    def __init__(self, hidden_size:int, n_heads:int, ff_hidden:int)->None:
        self.layer_norm_1 = LayerNorm(hidden_size)
        self.layer_norm_2 = LayerNorm(hidden_size)
        self.layer_norm_3 = LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, n_heads)
        self.cross_attention = MultiHeadAttention(hidden_size, n_heads)
        self.feed_forward = FeedForward(hidden_size, ff_hidden)

    def __call__(self, trg_seq:np.array, src_mask:np.array=None, trg_mask:np.array=None)->np.array:
        self_attn = self.self_attention(trg_seq, trg_seq, trg_seq, trg_seq, trg_mask)
        query = self.layer_norm_1(self_attn + trg_seq)
        cross_attn = self.cross_attention(query, trg_seq, trg_seq, src_mask)
        norm_2 = self.layer_norm_2(cross_attn + query)
        ff = self.feed_forward(norm_2)
        output = self.layer_norm_3(ff+norm_2)
        return output


class TransformerDecoder:
    def __init__(self, vocab_size:int, max_seq_len:int, hidden_size:int, n_heads:int, ff_hidden:int)->None:
        self.embedding = PositionalEmbedding(vocab_size, hidden_size, max_seq_len)
        self.blocks = [
            TransformerDecoderBlock(hidden_size,n_heads,ff_hidden) for _ in range(n_heads)
        ]
        self.linear = Linear(hidden_size, hidden_size)

    def __call__(self, trg_seq:np.array, src_mask:np.array, trg_mask:np.array)->np.array:
        embedded_seq = self.embedding(trg_seq)
        decoder_out = None
        for block in self.blocks:
            decoder_out = block(embedded_seq, src_mask, trg_mask)
        return decoder_out


class Transformer:
    def __init__(self, vocab_size:int, max_seq_len:int, hidden_size:int, n_heads:int, ff_hidden:int)->None:
        self.encoder = TransformerEncoder(vocab_size, max_seq_len, hidden_size, n_heads, ff_hidden)
        self.decoder = TransformerDecoder(vocab_size, max_seq_len, hidden_size, n_heads, ff_hidden)
        self.dropout = Dropout()
        self.classifier = Linear(hidden_size, vocab_size)

    def __call__(self, src_seq:np.array, trg_seq:np.array, src_mask:np.array, trg_mask:np.array)->np.array:
        encoder_out = self.encoder(src_seq, src_mask)
        decoder_out = self.decoder(trg_seq, src_mask, trg_mask)

        x = self.dropout(x)
        x = self.classifier(x)
        return x


if __name__=="__main__":
    model = Transformer(vocab_size=5000, hidden_size=768, max_seq_len=200, n_heads=12, ff_hidden=2024)

    seq = np.zeros([32, 27], dtype=int)
    mask = np.tril(np.ones((seq.shape[-1], seq.shape[-1]), dtype=np.float32), k=0)
    print(model(seq, mask).shape)    
    print(mask.shape)