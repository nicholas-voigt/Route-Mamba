import torch
import torch.nn as nn

from model import components as mc


class Critic(nn.Module):
    def __init__(self, input_dim, embedding_dim, mamba_hidden_dim, mamba_layers,
                 dropout, mlp_ff_dim, mlp_embedding_dim):
        super(Critic, self).__init__()

        # State Encoder
        self.state_embedder = mc.StructuralEmbeddingNet(
            input_dim = input_dim,
            embedding_dim = embedding_dim
        )
        self.state_embedding_norm = nn.LayerNorm(embedding_dim)
        self.state_encoder = mc.BidirectionalMambaEncoder(
            mamba_model_size = embedding_dim,
            mamba_hidden_state_size = mamba_hidden_dim,
            dropout = dropout,
            mamba_layers = mamba_layers
        )
        self.state_encoder_norm = nn.LayerNorm(2 * embedding_dim)

        # Value Decoder
        self.value_decoder = mc.MLP(
            input_dim = 2 * embedding_dim,
            feed_forward_dim = mlp_ff_dim,
            embedding_dim = mlp_embedding_dim,
            dropout = dropout,
            output_dim = 1
        )

    def forward(self, state):
        """
        Forward pass of the Critic network.
        Args:
            state: Tensor of shape (B, N, 2) representing the coordinates of the nodes.
        Returns:
            value: Tensor of shape (B, 1) representing the estimated Q-value for the (state, action) pair.
        """
        # Encode the State
        state_embedding = self.state_embedder(state)  # (B, N, E)
        state_embedding = self.state_embedding_norm(state_embedding) # (B, N, E)

        state_encoding = self.state_encoder(state_embedding)  # (B, N, 2E)
        state_encoding = self.state_encoder_norm(state_encoding)  # (B, N, 2E)

        # Decode Q-Value
        q = self.value_decoder(state_encoding.mean(dim=1))  # (B, 1)
        return q

