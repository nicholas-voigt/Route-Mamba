import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import math
from mamba_ssm import Mamba


class EmbeddingNet(nn.Module):
    """
    Embedding Network for node features.
    Performs node feature embedding and cyclic positional encoding.
    """
    def __init__(self, input_dim: int, embedding_dim: int, num_harmonics: int, device: torch.device, alpha: float):
        super().__init__()
        self.node_feature_encoder = nn.Linear(input_dim, embedding_dim, bias=False)   # Linear layer for node feature embedding
        self.cyclic_projection = nn.Linear(2 * num_harmonics, embedding_dim, bias=False)  # Linear layer to project harmonics to E

        self.k = num_harmonics  # Number of harmonics
        self.alpha = alpha      # Scaling factor for frequency base
        self.device = device    # Device for computation

    def cyclic_encoding(self, N: int):
        """
        Cyclic embedding which incorporates relative positional information.
        Args:
            N: Number of positions (tour length)
        Returns:
            node feature embedding: A tensor of shape (B, N, E)
            cyclic embedding: A tensor of shape (B, N, E)
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
        x_norm = F.layer_norm(x, (x.shape[-1],))  # [B, N, I]
        nfe = self.node_feature_encoder(x_norm)  # [B, N, E]

        # Cyclic Embedding
        B, N, _ = x.shape
        ce = self.cyclic_encoding(N).unsqueeze(0).repeat(B, 1, 1)  # [B, N, 2K]
        ce = self.cyclic_projection(ce)  # [B, N, E]

        return nfe, ce


class MambaBlock(nn.Module):
    """
    Mamba Block for the TSP model.
    Takes concatenated node and cyclic embeddings as input and outputs a score for each node.
    """
    def __init__(self, mamba_model_size, mamba_hidden_state_size, mamba_layers):
        """
        Args:
            mamba_model_size: Model dimension for Mamba (d_model), defined by input size
            mamba_hidden_state_size: SSM state expansion factor (d_state)
            mamba_layers: Number of Mamba blocks stacked (min: 1)
        """
        super(MambaBlock, self).__init__()

        self.forward_block = nn.ModuleList([
            Mamba(
                d_model=mamba_model_size,         # Model dimension d_model
                d_state=mamba_hidden_state_size,  # SSM state expansion factor
                d_conv=4,                         # Local convolution width
                expand=2                          # Block expansion factor
            ).to('cuda') for _ in range(mamba_layers)
        ])

        self.backward_block = nn.ModuleList([
            Mamba(
                d_model=mamba_model_size,         # Model dimension d_model
                d_state=mamba_hidden_state_size,  # SSM state expansion factor
                d_conv=4,                         # Local convolution width
                expand=2                          # Block expansion factor
            ).to('cuda') for _ in range(mamba_layers)
        ])

    def forward(self, x):
        """
        Args:
            x: (B, N, 2E)
        Returns:
            node_feats: (B, N, 4E)
        """
        # --- Forward Pass ---
        x_fwd = x
        for mamba_layer in self.forward_block:
            x_fwd = mamba_layer(x_fwd)
        # --- Backward Pass ---
        x_bwd = torch.flip(x, dims=[1])  # Reverse sequence
        for mamba_layer in self.backward_block:
            x_bwd = mamba_layer(x_bwd)
        x_bwd = torch.flip(x_bwd, dims=[1])  # Un-reverse to align
        # --- Concatenate ---
        return torch.cat([x_fwd, x_bwd], dim=-1)  # (B, N, 4E)


class BilinearScoreHead(nn.Module):
    """
    Takes Mamba and cyclic features as input and builds a score matrix S for gumbel-sinkhorn soft permutation.
    Build S in [B x N x N] from:
      E = mamba_features in [B x N x 4E]
      R = cyclic_feats    in [B x N x E]
      W = bilinear_weights in [d_head x d_head]
    via S = (E U_e) W (R U_r)^T
    """
    def __init__(self, model_vector_size: int, cycle_vector_size: int, score_head_dim: int = 128, bias: bool = True):
        super().__init__()
        self.proj_e = nn.Linear(model_vector_size, score_head_dim, bias=False)  # U_e
        self.proj_r = nn.Linear(cycle_vector_size, score_head_dim, bias=False) # U_r
        self.W      = nn.Parameter(torch.empty(score_head_dim, score_head_dim))    # bilinear core
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


class AttentionScoreHead(nn.Module):
    """
    Takes Mamba and cyclic features as input and builds a score matrix S for gumbel-sinkhorn soft permutation.
    Build S in [B x N x N] using multi-head attention from the Mamba features in [B x N x 4E].
    Score S[i, j] = attention score of node i attending to position j.
    """
    def __init__(self, model_vector_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        # Multi-head attention layer (pyTorch implementation), query=nodes, key=positions
        self.attention = nn.MultiheadAttention(
            embed_dim=model_vector_size, 
            num_heads=num_heads, 
            batch_first=True
        )
    
    def forward(self, mamba_features: torch.Tensor) -> torch.Tensor:
        """
        Perform self attention on Mamba features to get score matrix S.
        Args:
            mamba_features: [B, N, model_dim] - rich node features including context
        Returns:
            S: [B, N, N] (rows = nodes i, cols = positions j)
        """
        _, attn_weights = self.attention(
            query=mamba_features, 
            key=mamba_features, 
            value=mamba_features,
            need_weights=True,
            average_attn_weights=True
        )
        return attn_weights  # [B, N, N] (rows = nodes i, cols = positions j)
    

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
        Forward pass for the permutation layer. Converts a score matrix into a soft permutation matrix.
        Args:
            score matrix [B, N, N] - should be in logit space (unbounded)
        Returns:
            soft permutation matrix [B, N, N]
        """
        # Add Gumbel noise
        gumbel_noise = self.sample_gumbel(scores.shape, device=scores.device)
        scores = (scores + gumbel_noise) / self.gs_tau
        # Sinkhorn normalization to get doubly stochastic matrix
        soft_perm = self.sinkhorn(scores)
        return soft_perm
    

class TourConstructor(nn.Module):
    """
    Creates a hard tour from the given soft permutation matrix.
    Applies straight-through estimation for gradient flow.
    """
    def __init__(self, method: str):
        super(TourConstructor, self).__init__()
        self.method = method

    # ---- Greedy Permutation Hardening ----
    def greedy_hard_perm(self, soft_perm: torch.Tensor) -> torch.Tensor:
        """
        Converts a soft permutation matrix to a hard one using an efficient,
        iterative greedy assignment strategy that is fully vectorized across the batch.
        In each step, it finds the highest-scoring available assignment and commits to it.
        Args:
            soft_perm: (tensor: B, N, N) soft permutation matrix - rows = nodes, cols = positions
        Returns:
            (tensor: B, N, N) hard permutation matrix
        """
        B, N, _ = soft_perm.shape
        device, dtype = soft_perm.device, soft_perm.dtype

        row_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        col_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        NEG = torch.finfo(dtype).min
        hard_perm = torch.zeros_like(soft_perm)
        scores = soft_perm.clone()
        batch_indices = torch.arange(B, device=device)

        for _ in range(N):
            # Mask out scores of already assigned rows and columns via broadcasting
            scores.masked_fill_(row_mask.unsqueeze(2), NEG)
            scores.masked_fill_(col_mask.unsqueeze(1), NEG)
            # Find the entry with the highest score in the entire remaining matrix for each batch item
            _, flat_indices = scores.view(B, -1).max(dim=1)
            # Convert the flat index back to 2D row and column indices
            row_idx = flat_indices // N
            col_idx = flat_indices % N
            # Assign the winning entry in the hard permutation matrix
            hard_perm[batch_indices, row_idx, col_idx] = 1.0
            # Update the masks to mark this row and column as assigned
            row_mask[batch_indices, row_idx] = True
            col_mask[batch_indices, col_idx] = True

        return hard_perm
    
    # ---- Hungarian Permutation Hardening ----
    def hungarian_hard_perm(self, soft_perm: torch.Tensor) -> torch.Tensor:
        """
        Converts soft permutation matrix to hard permutation matrix using Hungarian algorithm.
        Runs on CPU only in O(N^3) time, but delivers best quality.
        Args:
            soft permutation matrix (B, N, N) - rows = nodes (i), cols = positions (j)
        """
        B, _, _ = soft_perm.shape
        hard_perm = torch.zeros_like(soft_perm)
        soft_perm_cpu = soft_perm.detach().cpu().numpy()  # Hungarian works on CPU only
        for b in range(B):
            # Maximize sum of P_ij  ==  minimize cost = negative P
            ri, cj = linear_sum_assignment(-soft_perm_cpu[b])
            hard_perm[b, ri, cj] = 1.0
        return hard_perm

    def forward(self, soft_perm: torch.Tensor):
        """
        Forward pass for the tour constructor. Converts a soft permutation matrix to hard permutation matrix using straight-through estimator.
        Forward pass uses hard permutation, backward pass uses soft permutation for gradient flow.
        Implements a greedy argmax approach and the hungarian algorithm.
        Args:
            soft_perm: (B, N, N) - soft permutation matrix, Rows = nodes (i), Cols = positions (j).
            method: (str) - method for hard permutation extraction ("greedy" or "hungarian")
        Returns:
            straight_through_perm: (B, N, N) - straight-through permutation matrix for gradient flow
        """
        if self.method == "greedy":
            hard_perm = self.greedy_hard_perm(soft_perm)
        elif self.method == "hungarian":
            hard_perm = self.hungarian_hard_perm(soft_perm)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        # ST trick for gradient flow
        straight_through_perm = hard_perm + (soft_perm - soft_perm.detach())
        return straight_through_perm
