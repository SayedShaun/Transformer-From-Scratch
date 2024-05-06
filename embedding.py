#This is the Class to Implement Embedding Layer
import torch

class Embedding(torch.nn.Module):
    def __init__(self, num_embeddings:int, embedding_dim:int) -> None:
        super(Embedding, self).__init__()
        self.weights = torch.randn(num_embeddings, embedding_dim)

    def forward(self, input_seq):
        if input_seq.dtype != torch.long:
            return input_seq == input_seq.long()
        return self.weights[input_seq] 
    
my_embed = Embedding(num_embeddings=5000, embedding_dim=500)
nn_embed = torch.nn.Embedding(num_embeddings=5000, embedding_dim=500)
seq = torch.ones(32, 15, dtype=torch.long)
print(my_embed(seq).shape, nn_embed(seq).shape)
        



