import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from mamba_ssm import Mamba


class EmbeddingNet(nn.Module):
    """
    Embedding Network for node features.
    Performs node feature embedding and cyclic positional encoding.
    """
    def __init__(self, node_dim: int, embedding_dim: int, k: int, seq_length: int, device: torch.device, alpha: float, freq_spread: float):
        super().__init__()
        self.node_feature_encoder = nn.Linear(node_dim, embedding_dim, bias=False)  # Linear layer for node feature embedding
        self.cyclic_projection = nn.Linear(2 * k, embedding_dim, bias=False)  # Linear layer to project harmonics to embedding_dim

        self.node_dim = node_dim            # Dimension of the initial node features
        self.embedding_dim = embedding_dim  # Dimension of the embedding space
        self.k = k                          # Number of harmonics
        self.alpha = alpha                  # Scaling factor for frequency base
        self.device = device                # Device for computation

    def cyclic_encoding(self, N: int):
        """
        Cyclic embedding which incorporates relative positional information.
        Args:
            N: Number of positions (tour length)
        Returns:
            node feature embedding: A tensor of shape (batch, N, embedding_dim)
            cyclic embedding: A tensor of shape (batch, N, embedding_dim)
        """
        # tour phases: positions 0...N-1 [N]
        t = torch.arange(N, device=self.device, dtype=torch.float32)
        # integer harmonics for exact periodicity [k]
        h = torch.arange(1, self.k + 1, device=self.device, dtype=torch.float32)
        # map angles on radian
        angles = 2 * math.pi * (t[:, None] * h[None, :] / N)  # [N, k]
        # interleave sin/cos & decay amplitudes for higher frequencies
        emb = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).reshape(N, 2 * self.k)  # [N, k, 2] -> [N, 2k]
        a = h.pow(-self.alpha)
        emb = emb * a.repeat_interleave(2).unsqueeze(0)  # [N, 2k]
        return emb

    def forward(self, x):
        """
        Args:
            x: (batch, N, node_dim) - node features
        Returns:
            feats: (batch, N, embedding_dim * 2) - concatenated node and cyclic embeddings
        """
        # Node Feature Embedding
        x_norm = F.layer_norm(x, (x.shape[-1],))  # [B, N, node_dim]
        nfe = self.node_feature_encoder(x_norm)  # [B, N, embedding_dim]

        # Cyclic Embedding
        B, N, _ = x.shape
        ce = self.cyclic_encoding(N).unsqueeze(0).repeat(B, 1, 1)  # [B, N, 2k]
        ce = self.cyclic_projection(ce)  # [B, N, embedding_dim]

        return nfe, ce


class MambaBlock(nn.Module):
    """
    Mamba Block for the TSP model.
    Takes concatenated node and cyclic embeddings as input and outputs a score for each node.
    """
    def __init__(self, input_dim, mamba_dim, hidden_dim, score_dim):
        """
        Args:
            input_dim: Dimension of concatenated embedding (node + cyclic)
            mamba_dim: Model dimension for Mamba (d_model)
            hidden_dim: SSM state expansion factor (d_state)
            score_dim: Dimension of the output score vector
        """
        super(MambaBlock, self).__init__()
        self.embedding_dim = input_dim  # Dimension of the input embeddings
        self.mamba_dim = mamba_dim if mamba_dim else input_dim  # Mamba model dimension (d_model)
        self.hidden_dim = hidden_dim    # Hidden dimension for Mamba
        self.score_dim = score_dim      # Output score dimension

        self.input_proj = nn.Linear(self.embedding_dim, self.mamba_dim) if self.embedding_dim != self.mamba_dim else None

        self.mamba = Mamba(
            d_model=self.mamba_dim,   # Model dimension d_model
            d_state=self.hidden_dim,  # SSM state expansion factor
            d_conv=4,                 # Local convolution width
            expand=2                  # Block expansion factor
        ).to("cuda")

        self.scorer = nn.Linear(self.mamba_dim, self.score_dim)  # Multidimensional scores

    def forward(self, x):
        """
        Args:
            x: (batch, N, embedding_dim)
        Returns:
            scores: (batch, N, score_dim)
        """
        if self.input_proj is not None: # Project input to Mamba dimension (safety)
            x = self.input_proj(x)
        h = self.mamba(x)  # (batch, N, embedding_dim)
        scores = self.scorer(h)  # (batch, N, score_dim)
        return scores


class ValueDecoder(nn.Module):
    """
    Value Decoder for the Mamba model.
    Takes output score vectors from Mamba and performs gumbel sinkhorn sorting to generate a new tour.
    """
    def __init__(self, score_dim, gs_tau, gs_iters):
        super(ValueDecoder, self).__init__()
        self.score_dim = score_dim
        self.gs_tau = gs_tau
        self.gs_iters = gs_iters

        self.score_proj = nn.Linear(self.score_dim, 1)

    # ---- Gumbel noise sampling ----
    def sample_gumbel(self, shape, eps=1e-20, device=None):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    # ---- Sinkhorn normalization ----
    def sinkhorn(self, log_alpha):
        for _ in range(self.gs_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)  # row norm
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)  # col norm
        return torch.exp(log_alpha)

    # ---- Gumbel-Sinkhorn operator ----
    def gumbel_sinkhorn(self, scores):
        """
        Args:
            scores: (batch, N)
        Returns:
            (batch, N, N) permutation matrix
        """
        _, N = scores.shape
        # Expand scores to NxN matrix by repeating across columns
        log_alpha = scores.unsqueeze(2).expand(-1, -1, N)  # (batch, N, N)
        # Add Gumbel noise
        gumbel_noise = self.sample_gumbel(log_alpha.shape, device=scores.device)
        log_alpha = (log_alpha + gumbel_noise) / self.gs_tau
        # Sinkhorn normalization to get doubly stochastic matrix
        P_hat = self.sinkhorn(log_alpha)
        return P_hat  # (batch, N, N)

    def forward(self, scores):
        """
        Args:
            scores: (batch, N, score_dim) - scores from Mamba model
        Returns:
            P_hat: (batch, N, N) - soft permutation matrix
        """
        # Project multidimensional scores to scalar per node
        scalar_scores = self.score_proj(scores).squeeze(-1)  # (batch, N)
        # Apply Gumbel-Sinkhorn
        P_hat = self.gumbel_sinkhorn(scalar_scores)
        return P_hat