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
    def __init__(self, node_dim: int, embedding_dim: int, k: int, device: torch.device, alpha: float):
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
    def __init__(self, input_dim, mamba_dim, hidden_dim, layers):
        """
        Args:
            input_dim: Dimension of concatenated embedding (node + cyclic)
            mamba_dim: Model dimension for Mamba (d_model)
            hidden_dim: SSM state expansion factor (d_state)
            layers: Number of Mamba blocks stacked (min: 1)
        """
        super(MambaBlock, self).__init__()
        self.embedding_dim = input_dim  # Dimension of the input embeddings
        self.mamba_dim = mamba_dim if mamba_dim else input_dim  # Mamba model dimension (d_model)
        self.hidden_dim = hidden_dim    # Hidden dimension for Mamba

        self.input_proj = nn.Linear(self.embedding_dim, self.mamba_dim) if self.embedding_dim != self.mamba_dim else None

        self.mamba_layers = nn.ModuleList([
            Mamba(
                d_model=self.mamba_dim,   # Model dimension d_model
                d_state=self.hidden_dim,  # SSM state expansion factor
                d_conv=4,                 # Local convolution width
                expand=2                  # Block expansion factor
            ).to('cuda') for _ in range(layers)
        ])

    def forward(self, x):
        """
        Args:
            x: (batch, N, embedding_dim)
        Returns:
            node_feats: (batch, N, model_dim)
        """
        if self.input_proj is not None: # Project input to Mamba dimension (safety)
            x = self.input_proj(x)
        for mamba_block in self.mamba_layers:
            x = mamba_block(x)
        return x    # [B, N, mamba_dim]


class BilinearScoreHead(nn.Module):
    """
    Takes Mamba and cyclic features as input and builds a score matrix S for gumbel-sinkhorn soft permutation.
    Build S in [B x N x N] from:
      E = mamba_features in [B x N x model_dim]
      R = cyclic_feats    in [B x N x embedding_dim]
      W = bilinear_weights in [d_head x d_head]
    via S = (E U_e) W (R U_r)^T
    """
    def __init__(self, model_dim: int, embedding_dim: int, d_head: int = 128, bias: bool = True):
        super().__init__()
        self.proj_e = nn.Linear(model_dim,    d_head, bias=False)  # U_e
        self.proj_r = nn.Linear(embedding_dim, d_head, bias=False) # U_r
        self.W      = nn.Parameter(torch.empty(d_head, d_head))    # bilinear core
        nn.init.xavier_uniform_(self.W)

        if bias:
            self.node_bias = nn.Parameter(torch.zeros(1, 1, 1))   # optional global scalar
            self.pos_bias  = nn.Parameter(torch.zeros(1, 1, 1))
        else:
            self.register_parameter("node_bias", None)
            self.register_parameter("pos_bias", None)

    def forward(self, mamba_features: torch.Tensor, cyclic_feats: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mamba_features: [B, N, model_dim]
            cyclic_feats:   [B, N, embedding_dim]
        Returns:
            S: [B, N, N] (rows = nodes i, cols = positions j)
        """
        E = self.proj_e(mamba_features)          # [B, N, d_head]
        R = self.proj_r(cyclic_feats)            # [B, N, d_head]

        # Apply bilinear core: (E @ W) @ R^T
        Ew = E @ self.W                          # [B, N, d_head]
        S  = torch.matmul(Ew, R.transpose(1, 2)) # [B, N, N]

        if self.node_bias is not None:
            S = S + self.node_bias + self.pos_bias
        return S


class GumbelSinkhornDecoder(nn.Module):
    """
    Takes score matrix [B, N, N] as input,
    introduces Gumbel noise via sampling and scales scores according to temperature gs_tau,
    performs Sinkhorn normalization for gs_iters iterations (larger -> closer to true permutation) to get a doubly stochastic matrix.
    """
    def __init__(self, gs_tau, gs_iters):
        super().__init__()
        self.gs_tau = gs_tau
        self.gs_iters = gs_iters

    # ---- Gumbel noise sampling ----
    def sample_gumbel(self, shape, eps=1e-20, device=None):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    # ---- Sinkhorn normalization ----
    def sinkhorn(self, scores: torch.Tensor) -> torch.Tensor:
        for _ in range(self.gs_iters):
            scores = scores - torch.logsumexp(scores, dim=2, keepdim=True)  # row norm
            scores = scores - torch.logsumexp(scores, dim=1, keepdim=True)  # col norm
        return torch.exp(scores)

    def forward(self, scores: torch.Tensor) -> torch.Tensor:
        """
        Args:
            score matrix [B, N, N] - should be in logit space (unbounded)
        Returns:
            soft permutation matrix [B, N, N]
        """
        # Add Gumbel noise
        gumbel_noise = self.sample_gumbel(scores.shape, device=scores.device)
        scores = (scores + gumbel_noise) / self.gs_tau
        # Sinkhorn normalization to get doubly stochastic matrix
        perm = self.sinkhorn(scores)
        return perm