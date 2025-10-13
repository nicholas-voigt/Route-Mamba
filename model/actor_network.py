import torch
from torch import nn

from model import components as mc


class SinkhornPermutationActor(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_harmonics, frequency_scaling, mamba_hidden_dim, mamba_layers, 
                 num_attention_heads, ffn_expansion, initial_identity_bias, gs_tau, gs_iters, method, dropout):
        super().__init__()

        # Model components
        self.encoder = mc.EmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embedding_dim,
            num_harmonics = num_harmonics,
            alpha = frequency_scaling
        )
        self.embedding_norm = nn.LayerNorm(2 * embedding_dim)
        self.model = mc.BidirectionalMambaEncoder(
            mamba_model_size = 2 * embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )
        self.mamba_norm = nn.LayerNorm(4 * embedding_dim)
        self.score_constructor = mc.AttentionScoreHead(
            model_dim = 4 * embedding_dim,
            num_heads = num_attention_heads,
            ffn_expansion = ffn_expansion,
            dropout = dropout
        )
        self.identity_bias = nn.Parameter(torch.full((1,), initial_identity_bias))
        self.decoder = mc.GumbelSinkhornDecoder(
            gs_tau = gs_tau,
            gs_iters = gs_iters
        )
        self.tour_constructor = mc.TourConstructor(
            method = method
        )

    def forward(self, batch):
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
        Returns:
            st_perm: (B, N, N) - doubly stochastic matrix (soft permutation matrix)
            hard_perm: (B, N, N) - permutation matrix (hard assignment of the tour)
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
        identity_matrix = torch.eye(score_matrix.size(1), device=score_matrix.device) * self.identity_bias
        biased_score_matrix = score_matrix + identity_matrix

        # 6. Decoder Workshop 1: Use Gumbel-Sinkhorn to get soft permutation matrix (tour)
        soft_perm = self.decoder(biased_score_matrix)  # (B, N, N)

        # 7. Decoder Workshop 2: Get the straight-through permutation matrix by hard assignment
        hard_perm = self.tour_constructor(soft_perm)

        return soft_perm, hard_perm


class ARPointerActor(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_harmonics, frequency_scaling, mamba_hidden_dim, mamba_layers, dropout):
        super().__init__()

        # Model components
        self.feature_embedder = mc.EmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embedding_dim,
            num_harmonics = num_harmonics,
            alpha = frequency_scaling
        )
        self.embedding_norm = nn.LayerNorm(2 * embedding_dim)
        self.encoder = mc.BidirectionalMambaEncoder(
            mamba_model_size = 2 * embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )
        self.encoder_norm = nn.LayerNorm(4 * embedding_dim)
        self.decoder = mc.ARPointerDecoder(
            embedding_dim = 4 * embedding_dim,
            mamba_hidden_dim = 1024,
            key_proj_bias = True,
            dropout = dropout
        )


    def forward(self, batch):
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
        Returns:
            tour: (B, N) - node indices representing the tour
            logits: (B, ) - sum of log probabilities of the chosen actions
        """
        # 1. Create Embeddings, normalize and concatenate
        node_embeddings, cyclic_embeddings = self.feature_embedder(batch)  # (B, N, E), (B, N, E)
        embeddings = self.embedding_norm(torch.cat([node_embeddings, cyclic_embeddings], dim=-1))

        # 2. Encoding Workshop: Layered Mamba blocks with internal Pre-LN and external Post-LN
        encoded_features = self.encoder(embeddings)   # (B, N, 2M)
        encoded_features = self.encoder_norm(encoded_features)   # (B, N, 2M)
        encoded_graph = encoded_features.mean(dim=1)  # (B, 2M)

        # 3. Decoding Workshop: Autoregressive Pointer Network
        hard_perm, prob_dist = self.decoder(encoded_graph, encoded_features)  # (B, N, N), (B, N, N)

        return hard_perm, prob_dist
