import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, num_layers, dropout, **kwargs):
        super(LSTMModel, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.vocab_size = vocab_size

        self.embeddings = nn.Embedding(
            num_embeddings=self.vocab_size,
            embedding_dim=self.embedding_size
        )
        self.encoder = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout
            )

        self.head = nn.Linear(hidden_size, self.vocab_size)
    def forward(self, x):
        emb = self.embeddings(x)
        out, (h, c) = self.encoder(emb)
        out = self.head(out)
        return out