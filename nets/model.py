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
    def __init__(self, node_dim, embedding_dim, seq_length):
        super(EmbeddingNet, self).__init__()
        self.node_dim = node_dim  # Dimension of the initial node features
        self.embedding_dim = embedding_dim  # Dimension of the embedding space
        self.embedder = nn.Linear(node_dim, embedding_dim, bias=False)  # Linear layer for node feature embedding
        self.pattern = self.cyclic_encoding(seq_length, embedding_dim)  # Cyclic encoding

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def basesin(self, x, T, fai=0):
        return np.sin(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def basecos(self, x, T, fai=0):
        return np.cos(2 * np.pi / T * np.abs(np.mod(x, 2 * T) - T) + fai)

    def cyclic_encoding(self, n_position, emb_dim, mean_pooling=True):
        """
        Args:
            n_position: Number of positions (sequence length)
            emb_dim: Dimension of the embedding space
            mean_pooling: Whether to use mean pooling
        Returns:
            A tensor of shape (n_position, emb_dim) containing the cyclic encodings.
        """
        Td_set = np.linspace(np.power(n_position, 1 / (emb_dim // 2)), n_position, emb_dim // 2, dtype='int')
        x = np.zeros((n_position, emb_dim))
        for i in range(emb_dim):
            Td = Td_set[i // 3 * 3 + 1] if (i // 3 * 3 + 1) < (emb_dim // 2) else Td_set[-1]
            fai = 0 if i <= (emb_dim // 2) else 2 * np.pi * ((-i + (emb_dim // 2)) / (emb_dim // 2))
            longer_pattern = np.arange(0, np.ceil((n_position) / Td) * Td, 0.01)
            if i % 2 == 1:
                x[:, i] = self.basecos(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]
            else:
                x[:, i] = self.basesin(longer_pattern, Td, fai)[np.linspace(0, len(longer_pattern), n_position, dtype='int', endpoint=False)]
        pattern = torch.from_numpy(x).to(torch.float32)
        pattern_sum = torch.zeros_like(pattern)
        arange = torch.arange(n_position)
        pooling = [0] if not mean_pooling else [-2, -1, 0, 1, 2]
        time = 0
        for i in pooling:
            time += 1
            index = (arange + i + n_position) % n_position
            pattern_sum += pattern.gather(0, index.view(-1, 1).expand_as(pattern))
        pattern = 1. / time * pattern_sum - pattern.mean(0)
        return pattern

    def forward(self, x):
        """
        Args:
            x: (batch, N, node_dim) - node features
        Returns:
            feats: (batch, N, embedding_dim * 2) - concatenated node and cyclic embeddings
        """
        NFEs = self.embedder(x)  # (batch, N, embedding_dim)
        batch_size, N, _ = x.shape
        # Expand cyclic pattern for batch
        cyclic_emb = self.pattern.unsqueeze(0).expand(batch_size, -1, -1).to(x.device)  # (batch, N, embedding_dim)
        feats = torch.cat([NFEs, cyclic_emb], dim=-1)  # (batch, N, embedding_dim * 2)
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
    def __init__(self, score_dim, seq_length):
        super(ValueDecoder, self).__init__()
        self.score_dim = score_dim
        self.seq_length = seq_length

        self.score_proj = nn.Linear(self.score_dim, 1)

    # ---- Gumbel noise sampling ----
    def sample_gumbel(self, shape, eps=1e-20, device=None):
        U = torch.rand(shape, device=device)
        return -torch.log(-torch.log(U + eps) + eps)

    # ---- Sinkhorn normalization ----
    def sinkhorn(self, log_alpha, n_iters=20):
        for _ in range(n_iters):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=2, keepdim=True)  # row norm
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=1, keepdim=True)  # col norm
        return torch.exp(log_alpha)

    # ---- Gumbel-Sinkhorn operator ----
    def gumbel_sinkhorn(self, scores, tau=1.0, n_iters=20, hard=False):
        """
        Args:
            scores: (batch, N)
            tau: temperature for Gumbel noise
            n_iters: number of Sinkhorn iterations
            hard: if True, returns a hard permutation matrix (for inference), 
                if False, returns a soft permutation matrix (for training)
        Returns: 
            (batch, N, N) permutation matrix
        """
        batch_size, N = scores.shape
        # Expand scores to NxN matrix by repeating across columns
        log_alpha = scores.unsqueeze(2).expand(-1, -1, N)  # (batch, N, N)
        # Add Gumbel noise
        gumbel_noise = self.sample_gumbel(log_alpha.shape, device=scores.device)
        log_alpha = (log_alpha + gumbel_noise) / tau
        # Sinkhorn normalization to get doubly stochastic matrix
        P_hat = self.sinkhorn(log_alpha, n_iters=n_iters)
        if hard:
            # Convert to a true permutation matrix (no gradient)
            idx = torch.argmax(P_hat, dim=2)
            P_hard = torch.zeros_like(P_hat).scatter_(2, idx.unsqueeze(2), 1.0)
            P_hat = (P_hard - P_hat).detach() + P_hat
        return P_hat  # (batch, N, N)

    def forward(self, scores, tau=1.0, n_iters=20, hard=False):
        """
        Args:
            scores: (batch, N, score_dim) - scores from Mamba model
            tau: temperature for Gumbel noise
            n_iters: number of Sinkhorn iterations
            hard: if True, returns a hard permutation matrix (for inference)
        Returns:
            P_hat: (batch, N, N) - soft permutation matrix
        """
        # Project multidimensional scores to scalar per node
        scalar_scores = self.score_proj(scores).squeeze(-1)  # (batch, N)
        # Apply Gumbel-Sinkhorn
        P_hat = self.gumbel_sinkhorn(scalar_scores, tau=tau, n_iters=n_iters, hard=hard)
        return P_hat