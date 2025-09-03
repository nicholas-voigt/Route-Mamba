import torch
from torch import nn

from nets.model import EmbeddingNet, MambaBlock, GumbelSinkhornDecoder, BilinearScoreHead, TourConstructor


class Actor(nn.Module):
    def __init__(self, input_dim, embedding_dim, harmonics, frequency_scaling, model_dim, hidden_dim, mamba_layers, 
                 score_head_dim, score_head_bias, gs_tau, gs_iters, method, device):
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
            layers = mamba_layers
        )
        self.score_constructor = BilinearScoreHead(
            model_dim = model_dim,
            embedding_dim = embedding_dim,
            d_head = score_head_dim,
            bias = score_head_bias
        )
        self.decoder = GumbelSinkhornDecoder(
            gs_tau = gs_tau,
            gs_iters = gs_iters
        )
        self.tour_constructor = TourConstructor(
            method = method
        )

    def forward(self, batch):
        """
        Args:
            batch: (B, N, I) - node features
        Returns:
            st_perm: (B, N, N) - straight-through permutation matrix with hard assignments
        """
        # 1. Encode node features (and cyclic encoding)
        node_embeddings, cyclic_embeddings = self.encoder(batch)  # (B, N, embedding_dim), (B, N, embedding_dim)

        # 2. MambaBlock: get per-node score vectors
        mamba_feats = self.model(torch.cat([node_embeddings, cyclic_embeddings], dim=-1))   # (B, N, model_dim)

        # 3. ScoreHead: get score matrix (tour)
        score_matrix = self.score_constructor(mamba_feats, cyclic_embeddings)  # (B, N, N)

        # 4. ValueDecoder: get soft permutation matrix (tour)
        soft_perm = self.decoder(score_matrix)  # (B, N, N)

        # 5. Compute new tour via straight-through permutation
        new_tours = torch.bmm(self.tour_constructor(soft_perm), batch)  # (B, N, I)
        return new_tours