import torch
from torch import nn

from model import components as mc


class SinkhornPermutationActor(nn.Module):
    def __init__(self, input_dim, embedding_dim, kNN_neighbors, mamba_hidden_dim, mamba_layers, 
                 num_attention_heads, ffn_expansion, initial_identity_bias, gs_tau, gs_iters, method, dropout):
        super().__init__()

        # Model components
        self.feature_embedder = mc.StructuralEmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embedding_dim,
            k = kNN_neighbors
        )
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.encoder = mc.BidirectionalMambaEncoder(
            mamba_model_size = embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )
        self.encoder_norm = nn.LayerNorm(2 * embedding_dim)
        self.score_constructor = mc.AttentionScoreHead(
            model_dim = 2 * embedding_dim,
            num_heads = num_attention_heads,
            ffn_expansion = ffn_expansion,
            dropout = dropout
        )
        self.identity_bias = nn.Parameter(torch.full((1,), initial_identity_bias), requires_grad=True)
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
        # 1. Create Embeddings & normalize
        embeddings = self.feature_embedder(batch)  # (B, N, E)
        embeddings = self.embedding_norm(embeddings)  # (B, N, E)

        # 2. Encoder: Layered Mamba blocks with internal Pre-LN
        encoded_features = self.encoder(embeddings)
        encoded_features = self.encoder_norm(encoded_features)   # (B, N, 2E)

        # 3. Score Construction: Multi-Head Attention with FFN and internal Pre-LN and projection to scores + identity bias
        score_matrix = self.score_constructor(encoded_features)  # (B, N, N)
        identity_matrix = torch.eye(score_matrix.size(1), device=score_matrix.device) * self.identity_bias
        biased_score_matrix = score_matrix + identity_matrix

        # 4. Decoder Workshop: Use Gumbel-Sinkhorn to get soft permutation matrix & hard assignment via tour construction
        soft_perm = self.decoder(biased_score_matrix)  # (B, N, N)
        hard_perm = self.tour_constructor(soft_perm)

        return soft_perm, hard_perm


class ARPointerActor(nn.Module):
    def __init__(self, input_dim, embedding_dim, mamba_hidden_dim, mamba_layers, dropout):
        super().__init__()

        # Model components
        self.feature_embedder = nn.Linear(input_dim, embedding_dim, bias=False)
        self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.encoder = mc.BidirectionalMambaEncoder(
            mamba_model_size = embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )
        self.encoder_norm = nn.LayerNorm(2 * embedding_dim) # Because of bidirectional, output dim = 2 * input dim
        self.decoder = mc.ARPointerDecoder(
            embedding_dim = 2 * embedding_dim,
            mamba_hidden_dim = 256,
            key_proj_bias = False,
            dropout = dropout
        )


    def forward(self, batch):
        """
        Args:
            batch: (B, N, I) - node features with 2D coordinates
        Returns:
            tour: (B, N, N) - permutation matrix representing the tour
            prob_dist: (B, N, N) - probability distribution over next nodes at each step
        """
        # 1. Create Embeddings & normalize
        embeddings = self.embedding_norm(self.feature_embedder(batch))  # (B, N, E)

        # 2. Encoding Workshop: Layered Mamba blocks with internal Pre-LN and external Post-LN
        encoded_features = self.encoder(embeddings)   # (B, N, 2M)
        encoded_features = self.encoder_norm(encoded_features)   # (B, N, 2M)
        encoded_graph = encoded_features.mean(dim=1)  # (B, 2M)

        # 3. Decoding Workshop: Autoregressive Pointer Network
        hard_perm, prob_dist = self.decoder(encoded_graph, encoded_features)  # (B, N, N), (B, N, N)

        return hard_perm, prob_dist
