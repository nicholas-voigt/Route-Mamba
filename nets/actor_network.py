import torch
from torch import nn

from nets.model import EmbeddingNet, MambaBlock, GumbelSinkhornDecoder, AttentionScoreHead, TourConstructor


class Actor(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_harmonics, frequency_scaling, mamba_hidden_dim, mamba_layers, 
                 num_attention_heads, gs_tau, gs_iters, method):
        super().__init__()

        self.encoder = EmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embedding_dim,
            num_harmonics = num_harmonics,
            alpha = frequency_scaling
        )
        self.model = MambaBlock(
            mamba_model_size = 2 * embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            mamba_layers = mamba_layers
        )
        self.score_constructor = AttentionScoreHead(
            model_dim = 4 * embedding_dim,
            num_heads = num_attention_heads,
            ffn_expansion = 4,
            dropout = 0.1,
            num_layers = 2
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
            batch: (B, N, I) - node features with 2D coordinates
        Returns:
            st_perm: (B, N, I) - new tours
        """
        # 1. Encode node features (and cyclic encoding)
        node_embeddings, cyclic_embeddings = self.encoder(batch)  # (B, N, E), (B, N, E)

        # 2. MambaBlock: get per-node score vectors
        mamba_feats = self.model(torch.cat([node_embeddings, cyclic_embeddings], dim=-1))   # (B, N, 2M)

        # 3. ScoreHead: get score matrix (tour)
        score_matrix = self.score_constructor(mamba_feats)  # (B, N, N)

        # 4. ValueDecoder: get soft permutation matrix (tour)
        soft_perm = self.decoder(score_matrix)  # (B, N, N)

        # 5. Get the straight-through permutation matrix
        st_perm = self.tour_constructor(soft_perm)

        # 6. Compute new tour via straight-through permutation
        new_tours = torch.bmm(st_perm.transpose(1, 2), batch)  # (B, N, I)
        return new_tours