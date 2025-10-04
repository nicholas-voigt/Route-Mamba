import torch
import torch.nn as nn

from model.components import EmbeddingNet, BidirectionalMambaEncoder, ConvolutionBlock, MLP


class Critic(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_harmonics, frequency_scaling, mamba_hidden_dim, mamba_layers, dropout,
                 conv_out_channels, conv_kernel_size, conv_stride, mlp_ff_dim, mlp_embedding_dim):
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

        # Action Encoder
        self.action_encoder = nn.Sequential(
            ConvolutionBlock(in_channels=1, out_channels=conv_out_channels//2, kernel_size=conv_kernel_size, stride=conv_stride),
            ConvolutionBlock(in_channels=conv_out_channels//2, out_channels=conv_out_channels, kernel_size=conv_kernel_size, stride=conv_stride),
            nn.AdaptiveAvgPool2d(1)
        )

        # Value Decoder
        self.value_decoder = MLP(
            input_dim = 4 * embedding_dim + conv_out_channels,
            feed_forward_dim = mlp_ff_dim,
            embedding_dim = mlp_embedding_dim,
            output_dim = 1
        )

    def forward(self, state, action):
        # --- Process State ---
        # 1. Create Node and Cyclic Embeddings, Concatenate and Normalize
        node_embeddings, cyclic_embeddings = self.state_embedder(state)  # (B, N, E), (B, N, E)
        state_embedding = self.state_embedding_norm(torch.cat([node_embeddings, cyclic_embeddings], dim=-1))
        # 2. Mamba Workshop: Layered Mamba blocks with internal Pre-LN
        state_embedding = self.state_encoder(state_embedding)   # (B, N, 2M)
        state_embedding = state_embedding.mean(dim=1)  # (B, 2M)

        # --- Process Action ---
        # 1. Reshape Action to include channel dimension
        action = action.unsqueeze(1)  # (B, 1, N, N)
        # 2. Convolutional Action Encoder: Apply Convolutional Blocks and Global Pooling
        action_embedding = self.action_encoder(action)  # (B, C, 1, 1)
        action_embedding = action_embedding.view(action_embedding.size(0), -1)  # (B, C)

        # --- Decode Value ---
        # 1. Concatenate State and Action Embeddings
        q_embedding = torch.cat([state_embedding, action_embedding], dim=1)  # (B, 2M + C)
        # 2. Value Decoder: MLP to produce scalar value
        q = self.value_decoder(q_embedding)  # (B, 1)
        return q

