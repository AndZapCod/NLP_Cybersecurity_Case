import torch
import torch.nn as nn


class BiLSTMCRF(nn.Module):

    def init_weigths(self):
        for param in self.parameters():
            nn.init.normal_(param.data, mean=0, std=0.1)

    def init_embeddings(self):
        self.embedding.weight.data[
            self.padtok_idx
        ] = torch.zeros(self.embedding_size)

    def __init__(self,
                 input_size, hidden_size, num_clasess, padtok_idx,
                 num_layers=2,
                 embedding_size=100,
                 dropout=0):
        super(BiLSTMCRF, self).__init__()

        self.embedding_size = embedding_size
        self.padtok_idx = padtok_idx

        # Layers
        # Word embedding
        self.embedding = nn.Embedding(
            num_embeddings=input_size,
            embedding_dim=embedding_size,
            padding_idx=padtok_idx,
            max_norm=1.
        )
        self.emb_dropout = nn.Dropout(dropout)

        # BiLSTM Layer

        self.bilstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully Connected Layer

        self.fully_dropout = nn.Dropout(dropout)
        self.fully_connected = nn.Linear(hidden_size * num_layers, num_clasess)

        self.init_weigths()
        self.init_embeddings()

    def forward(self, sentence):
        embedding = self.emb_dropout(self.embedding(sentence))
        bilstm_out = self.bilstm(embedding)[0]
        output = self.fully_connected(self.fully_dropout(bilstm_out))
        return output
