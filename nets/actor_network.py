import torch
from torch import nn

from nets.model import EmbeddingNet, MambaBlock, ValueDecoder


class Actor(nn.Module):
    def __init__(self, input_dim, embedding_dim, harmonics, frequency_scaling, model_dim, hidden_dim, score_dim, gs_tau, gs_iters, device):
        super().__init__()

        self.encoder = EmbeddingNet(
            node_dim = input_dim,
            embedding_dim = embedding_dim,
            k = harmonics,
            device = device,
            alpha = frequency_scaling
        )
        self.model = MambaBlock(
            input_dim = embedding_dim * 2,
            mamba_dim = model_dim,
            hidden_dim = hidden_dim,
            score_dim = score_dim
        )
        self.decoder = ValueDecoder(
            score_dim = score_dim,
            gs_tau = gs_tau,
            gs_iters = gs_iters
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
        nfe, ce = self.encoder(batch)  # (batch_size, seq_length, embedding_dim * 2)

        # 2. MambaBlock: get per-node score vectors
        scores = self.model(torch.cat([nfe, ce], dim=-1))   # (batch_size, seq_length, score_dim)

        # 3. ValueDecoder: get soft permutation matrix (tour)
        soft_perm = self.decoder(scores)  # (batch_size, seq_length, seq_length)

        # 4. Compute soft tour by multiplying permutation with node coordinates
        # batch: (batch_size, seq_length, input_dim)
        soft_tour = torch.bmm(soft_perm, batch)  # (batch_size, seq_length, input_dim)

        return soft_tour, soft_perm