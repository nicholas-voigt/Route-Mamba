import torch
import torch.nn as nn

from model.components import EmbeddingNet, BidirectionalMambaEncoder, MLP


class Critic(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_harmonics, frequency_scaling, mamba_hidden_dim, mamba_layers,
                 dropout, mlp_ff_dim, mlp_embedding_dim):
        super(Critic, self).__init__()

        # State Encoder
        self.state_embedder = nn.Linear(input_dim, embedding_dim, bias=False)
        self.state_embedding_norm = nn.BatchNorm1d(embedding_dim)
        self.state_encoder = BidirectionalMambaEncoder(
            mamba_model_size = embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )
        self.state_encoder_norm = nn.BatchNorm1d(2 * embedding_dim)

        # Value Decoder
        self.value_decoder = MLP(
            input_dim = 2 * embedding_dim,
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
        state_embedding = self.state_embedder(state)  # (B, N, E)
        state_embedding = self.state_embedding_norm(state_embedding.permute(0, 2, 1)).permute(0, 2, 1) # (B, N, E)
        state_encoding = self.state_encoder_norm(self.state_encoder(state_embedding))  # (B, N, 2E)

        # Fuse State and Action
        expected_tours = torch.bmm(action.transpose(1, 2), state_encoding)  # (B, N, 2E)

        # Decode Q-Value
        q = self.value_decoder(expected_tours.mean(dim=1))  # (B, 1)
        return q

