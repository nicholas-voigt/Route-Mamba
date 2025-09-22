import torch
from torch import nn

from nets.model import EmbeddingNet, BidirectionalMambaEncoder, GumbelSinkhornDecoder, AttentionScoreHead, TourConstructor


class Actor(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_harmonics, frequency_scaling, mamba_hidden_dim, mamba_layers, 
                 num_attention_heads, gs_tau, gs_iters, method):
        super().__init__()

        # Model components
        self.encoder = EmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embedding_dim,
            num_harmonics = num_harmonics,
            alpha = frequency_scaling
        )
        self.embedding_norm = nn.LayerNorm(2 * embedding_dim)
        self.model = BidirectionalMambaEncoder(
            mamba_model_size = 2 * embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = 0.1,
            mamba_layers = mamba_layers
        )
        self.mamba_norm = nn.LayerNorm(4 * embedding_dim)
        self.score_constructor = AttentionScoreHead(
            model_dim = 4 * embedding_dim,
            num_heads = num_attention_heads,
            ffn_expansion = 4,
            dropout = 0.1
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
        # 1. Create Embeddings
        node_embeddings, cyclic_embeddings = self.encoder(batch)  # (B, N, E), (B, N, E)

        # 2. External Normalization and Concatenation: Prepare Input to Mamba
        total_embeddings = self.embedding_norm(torch.cat([node_embeddings, cyclic_embeddings], dim=-1))

        # 3. Mamba Workshop: Layered Mamba blocks with internal Pre-LN
        mamba_feats = self.model(total_embeddings)   # (B, N, 2M)

        # 4. External Normalization: Prepare Input to ScoreHead
        norm_mamba_feats = self.mamba_norm(mamba_feats)   # (B, N, 2M)

        # 5. Attention Workshop: Multi-Head Attention with FFN and internal Pre-LN and projection to scores
        score_matrix = self.score_constructor(norm_mamba_feats)  # (B, N, N)

        # 6. Decoder Workshop 1: Use Gumbel-Sinkhorn to get soft permutation matrix (tour)
        soft_perm = self.decoder(score_matrix)  # (B, N, N)

        # 7. Decoder Workshop 2: Get the straight-through permutation matrix by hard assignment
        st_perm = self.tour_constructor(soft_perm)

        # 8. Compute new tour via straight-through permutation
        new_tours = torch.bmm(st_perm.transpose(1, 2), batch)  # (B, N, I)
        return new_tours