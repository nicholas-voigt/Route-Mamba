import torch
from torch import nn

from model import EmbeddingNet, MambaBlock, ValueDecoder


class Actor(nn.Module):
    def __init__(self, input_dim, embedding_dim, model_dim, hidden_dim, score_dim, gs_tau, gs_iters, seq_length):
        super().__init__()

        self.input_dim = input_dim          # Node input dimensions (usually 2 for (x, y) coordinates)
        self.embedding_dim = embedding_dim  # Embedding dimensions for node features and cyclic positional encodings (has to be even)
        self.seq_length = seq_length        # Sequence length (number of nodes in the graph)

        self.model_dim = model_dim          # Model dimensions for the MambaBlock
        self.hidden_dim = hidden_dim        # Hidden state dimensions for the MambaBlock
        self.score_dim = score_dim          # Score dimensions for the output vector

        self.gs_tau = gs_tau                # Gumbel-Sinkhorn temperature
        self.gs_iters = gs_iters            # Number of Sinkhorn iterations

        # Define model
        self.encoder = EmbeddingNet(
            self.input_dim,
            self.embedding_dim,
            self.seq_length
        )

        self.model = MambaBlock(
            self.embedding_dim * 2,
            self.model_dim,
            self.hidden_dim,
            self.score_dim
        )

        self.decoder = ValueDecoder(
            self.score_dim,
            self.gs_tau,
            self.gs_iters
        )

    def forward(self, batch):
        """
        Args:
            batch: (batch_size, seq_length, input_dim) - node features
        Returns:
            soft_tour: (batch_size, seq_length, input_dim) - soft tour (weighted sum of node features)
            soft_perm: (batch_size, seq_length, seq_length) - soft permutation matrix (tour)
        """
        # 1. Encode node features (and cyclic encoding)
        embeddings = self.encoder(batch)  # (batch_size, seq_length, embedding_dim * 2)

        # 2. MambaBlock: get per-node score vectors
        scores = self.model(embeddings)   # (batch_size, seq_length, score_dim)

        # 3. ValueDecoder: get soft permutation matrix (tour)
        soft_perm = self.decoder(scores)  # (batch_size, seq_length, seq_length)

        # 4. Compute soft tour by multiplying permutation with node coordinates
        # batch: (batch_size, seq_length, input_dim)
        soft_tour = torch.bmm(soft_perm, batch)  # (batch_size, seq_length, input_dim)

        return soft_tour, soft_perm