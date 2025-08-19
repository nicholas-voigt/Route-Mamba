import torch
import torch.nn as nn
import math
import numpy as np
from mamba_ssm import Mamba


class EmbeddingNet(nn.Module):
    """
    Embedding Network for node features.
    Performs node feature embedding and cyclic positional encoding.
    """
    def __init__(self, node_dim, embedding_dim, seq_length, device, alpha, freq_spread):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim            # Dimension of the initial node features
        self.embedding_dim = embedding_dim  # Dimension of the embedding space
        self.alpha = alpha                  # Scaling factor for frequency base
        self.freq_spread = freq_spread      # Density of cyclic encoding
        self.seq_length = seq_length        # Length of the sequence (tour length)
        self.device = device                # Device for computation

        self.node_feature_encoder = nn.Linear(node_dim, embedding_dim, bias=False)  # Linear layer for node feature embedding
        self.cyclic_encoder = self.cyclic_encoding(seq_length, embedding_dim, alpha, freq_spread)

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def cyclic_encoding(self, N: int, d: int, alpha: float, spread: float):
        """
        Cyclic embedding which incorporates relative positional information.
        Args:
            N: Number of positions (tour length)
            d: Dimension of the embedding space (has to be even)
            alpha: scaling factor for frequency base (theta = alpha * N)
            spread: frequency bandwith, between 1.0 (dense encoding) and 2.0 (standard RoPE)
        Returns:
            embedding: A tensor of shape (N, d) containing the cyclic encodings.
        """
        assert d % 2 == 0, "Embedding dimension must be even."
        assert 1.0 <= spread <= 2.0, "spread must be between 1.0 and 2.0"
        theta = alpha * N  # scaling factor for frequency base
        K = d // 2
        # tour phases: positions 0...N-1
        t = torch.arange(N, device=self.device, dtype=torch.float32)
        # Generate frequency spectrum
        freqs = 1.0 / (theta ** (torch.arange(0, K, 1, device=self.device, dtype=torch.float32) / K * spread))
        # angles = phase * frequencies (N, K)
        angles = t[:, None] * freqs
        emb_sin = torch.sin(angles)
        emb_cos = torch.cos(angles)
        # Initialize embedding tensor and interleave sin and cos across dimensions
        emb = torch.zeros((N, d), device=self.device, dtype=torch.float32)
        emb[:, 0::2] = emb_sin
        emb[:, 1::2] = emb_cos
        # Normalize for stability
        emb = emb - emb.mean(dim=0, keepdim=True)
        return emb

    def forward(self, x):
        """
        Args:
            x: (batch, N, node_dim) - node features
        Returns:
            feats: (batch, N, embedding_dim * 2) - concatenated node and cyclic embeddings
        """
        NFEs = self.node_feature_encoder(x)  # (batch, N, embedding_dim)
        batch_size, N, _ = x.shape
        # Expand cyclic pattern for batch
        CEs = self.cyclic_encoder.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)  # (batch, N, embedding_dim)
        feats = torch.cat([NFEs, CEs], dim=-1)  # (batch, N, embedding_dim * 2)
        return feats


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