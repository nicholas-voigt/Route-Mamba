import torch
import torch.nn as nn

from model.components import EmbeddingNet, BidirectionalMambaEncoder, MLP


class Critic(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_harmonics, frequency_scaling, mamba_hidden_dim, mamba_layers,
                 dropout, mlp_ff_dim, mlp_embedding_dim):
        super(Critic, self).__init__()

        # State Encoder
        self.state_embedder = EmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embedding_dim,
            num_harmonics = num_harmonics,
            alpha = frequency_scaling
        )
        self.state_embedding_norm = nn.LayerNorm(2 * embedding_dim)
        self.state_encoder = BidirectionalMambaEncoder(
            mamba_model_size = 2 * embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )

        # Fused Tour Encoder
        self.tour_encoder = BidirectionalMambaEncoder(
            mamba_model_size = 4 * embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = 1
        )

        # Value Decoder
        self.value_decoder = MLP(
            input_dim = 8 * embedding_dim,
            feed_forward_dim = mlp_ff_dim,
            embedding_dim = mlp_embedding_dim,
            dropout = dropout,
            output_dim = 1
        )

    def forward(self, state, action):
        """
        Forward pass of the Critic network.
        Args:
            state: Tensor of shape (B, N, 2) representing the coordinates of the nodes.
            action: Tensor of shape (B, N, N) representing the permutation matrix of the action. rows = nodes, cols = positions in tour.
        Returns:
            value: Tensor of shape (B, 1) representing the estimated Q-value for the (state, action) pair.
        """
        # Encode the State
        node_embeddings, cyclic_embeddings = self.state_embedder(state)  # (B, N, E), (B, N, E)
        state_embedding = self.state_embedding_norm(torch.cat([node_embeddings, cyclic_embeddings], dim=-1))
        state_embedding = self.state_encoder(state_embedding)  # (B, N, 2M)

        # Fuse State and Action 
        expected_tours = torch.bmm(action.transpose(1, 2), state_embedding)  # (B, N, 2M)
        fused_embedding = self.tour_encoder(expected_tours)

        # Decode Q-Value
        q = self.value_decoder(fused_embedding.mean(dim=1))  # (B, 1)
        return q

