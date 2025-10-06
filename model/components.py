import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
import math
from mamba_ssm import Mamba


class EmbeddingNet(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int, num_harmonics: int, alpha: float):
        """
        Embedding Network for node features.
        Performs node feature embedding and cyclic positional encoding.
        Args:
            input_dim: Dimension of input node features (e.g., 2 for 2D coordinates)
            embedding_dim: Dimension of the output embeddings (node and cyclic respectively)
            num_harmonics: Number of harmonics for cyclic encoding (recommended: <= N/2)
            alpha: Scaling factor for frequency base
        """
        super(EmbeddingNet, self).__init__()
        self.node_feature_encoder = nn.Linear(input_dim, embedding_dim, bias=False)   # Linear layer for node feature embedding
        self.cyclic_projection = nn.Linear(2 * num_harmonics, embedding_dim, bias=False)  # Linear layer to project harmonics to Embedding space
        self.k = num_harmonics  # Number of harmonics
        self.alpha = alpha      # Scaling factor for frequency base

    def cyclic_encoding(self, N: int, device: torch.device) -> torch.Tensor:
        """
        Cyclic embedding which incorporates relative positional information.
        Args:
            N: Number of positions (tour length)
            device: Device to create the tensor on
        Returns:
            cyclic embedding: A tensor of shape (B, N, E)
        """
        # tour phases: positions 0...N-1 [N]
        t = torch.arange(N, device=device, dtype=torch.float32)
        # integer harmonics for exact periodicity [k]
        h = torch.arange(1, self.k + 1, device=device, dtype=torch.float32)
        # map angles on radian
        angles = 2 * math.pi * (t[:, None] * h[None, :] / N)  # [N, k]
        # interleave sin/cos & decay amplitudes for higher frequencies
        emb = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).reshape(N, 2 * self.k)  # [N, k, 2] -> [N, 2k]
        a = h.pow(-self.alpha)
        emb = emb * a.repeat_interleave(2).unsqueeze(0)  # [N, 2k]
        return emb

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch, N, node_dim) - node features
        Returns:
            feats: (batch, N, embedding_dim * 2) - concatenated node and cyclic embeddings
        """
        B, N, _ = x.shape
        # Node Feature Embedding
        nfe = self.node_feature_encoder(x)  # [B, N, E]
        # Cyclic Embedding
        ce = self.cyclic_encoding(N, x.device).unsqueeze(0).repeat(B, 1, 1)  # [B, N, 2K]
        ce = self.cyclic_projection(ce)  # [B, N, E]
        return nfe, ce


class MambaBlock(nn.Module):
    def __init__(self, mamba_model_size: int, mamba_hidden_state_size: int, dropout: float):
        """
        Single Mamba Block for the TSP model. Designed with reference to the Attention Encoder Layer.
        Applies Pre-LN: LayerNorm -> Mamba -> Dropout -> Residual
        Takes concatenated node and cyclic embeddings as input and outputs a score for each node. Does not change vector dimensions.
        Args:
            mamba_model_size: Model dimension for Mamba (d_model), defined by input size
            mamba_hidden_state_size: SSM state expansion factor (d_state)
            dropout: Dropout rate for regularization
        """
        super(MambaBlock, self).__init__()
        self.mamba = Mamba(
            d_model=mamba_model_size,
            d_state=mamba_hidden_state_size,
            d_conv=4,
            expand=2
        )
        self.norm = nn.LayerNorm(mamba_model_size)
        self.dropout = nn.Dropout(dropout)
        self.init_parameter_delta()  # initialize parameter delta as recommended in the Mamba paper

    def init_parameter_delta(self):
        # Initialize dt_proj bias as recommended in the Mamba paper
        dt_rank = self.mamba.dt_proj.weight.size(0)
        dt_init_std = dt_rank**-0.5 * 0.1
        # Initialize dt_proj weights
        nn.init.normal_(self.mamba.dt_proj.weight, mean=0.0, std=dt_init_std)
        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(dt_rank) * (math.log(0.1) - math.log(0.01)) + math.log(0.01)
        ).clamp(min=0.001)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.mamba.dt_proj.bias.copy_(inv_dt)
        # Initialize out_proj weights using xavier initialization
        nn.init.xavier_uniform_(self.mamba.out_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-Mamba-Normalization
        normalized_x = self.norm(x)
        # Mamba Layer
        mamba_output = self.mamba(normalized_x)
        # Residual connection
        x = x + self.dropout(mamba_output)
        return x


class BidirectionalMambaEncoder(nn.Module):
    def __init__(self, mamba_model_size: int, mamba_hidden_state_size: int, dropout: float, mamba_layers: int):
        """
        Mamba Implementation for the TSP model.
        Takes concatenated node and cyclic embeddings as input and outputs a score for each node.
        Args:
            mamba_model_size: Model dimension for Mamba (d_model), defined by input size
            mamba_hidden_state_size: SSM state expansion factor (d_state)
            dropout: Dropout rate for regularization
            mamba_layers: Number of Mamba blocks stacked (min: 1)
        """
        super(BidirectionalMambaEncoder, self).__init__()

        self.forward_block = nn.ModuleList([
            MambaBlock(mamba_model_size, mamba_hidden_state_size, dropout) for _ in range(mamba_layers)
        ])

        self.backward_block = nn.ModuleList([
            MambaBlock(mamba_model_size, mamba_hidden_state_size, dropout) for _ in range(mamba_layers)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, model_vector_size: int, cycle_vector_size: int, score_head_dim: int, bias: bool):
        """
        Takes Mamba and cyclic features as input and builds a score matrix S for gumbel-sinkhorn soft permutation.
        Build S in [B x N x N] from:
        E = mamba_features in [B x N x 4E]
        R = cyclic_feats    in [B x N x E]
        W = bilinear_weights in [d_head x d_head]
        via S = (E U_e) W (R U_r)^T
        """
        super(BilinearScoreHead, self).__init__()
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
    def __init__(self, model_dim: int, num_heads: int, ffn_expansion: int, dropout: float):
        """
        Transformer-style Multihead Attention Score Head for TSP.
        Takes Mamba features as input, processes them through one Transformer-style Encoder layer, including Feed-Forward Network with Pre-LN,
        and then projects them to a final score matrix S in [B x N x N].
        Args:
            model_dim: Dimension of the input features (d_model)
            num_heads: Number of attention heads
            ffn_expansion: Expansion factor for the Feed-Forward Network
            dropout: Dropout rate for regularization
        """
        super(AttentionScoreHead, self).__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=model_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(model_dim, model_dim * ffn_expansion),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * ffn_expansion, model_dim)
        )
        self.attn_norm = nn.LayerNorm(model_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.ffn_norm = nn.LayerNorm(model_dim)
        self.ffn_dropout = nn.Dropout(dropout)

        self.query_projection = nn.Linear(model_dim, model_dim)  # Final projection to score matrix dimension
        self.key_projection = nn.Linear(model_dim, model_dim)  # Final projection to score matrix dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform self attention on Mamba features to get score matrix S.
        Args:
            mamba_features: [B, N, model_dim] - rich node features including context
        Returns:
            Score Matrix: [B, N, N]
        """
        # Pre-Attention-Normalization
        normalized_x = self.attn_norm(x)
        # Multi-Head Self-Attention
        attn_output, _ = self.attention(normalized_x, normalized_x, normalized_x)
        # Residual connection
        x = x + self.attn_dropout(attn_output)
        # Pre-Feed-Forward-Normalization
        normalized_x = self.ffn_norm(x)
        # Feed-Forward Network
        ffn_output = self.ffn(normalized_x)
        # Residual connection
        x = x + self.ffn_dropout(ffn_output)
        # Feature projection to get query and key for score calculation
        query = self.query_projection(x)   # [B, N, model_dim]
        key = self.key_projection(x)     # [B, N, model_dim]
        # Score matrix calculation via dot product
        scores = torch.matmul(query, key.transpose(1, 2))  # [B, N, N]
        return scores


class GumbelSinkhornDecoder(nn.Module):
    def __init__(self, gs_tau: float, gs_iters: int):
        """
        Takes score matrix [B, N, N] as input,
        introduces Gumbel noise via sampling and scales scores according to temperature gs_tau,
        performs Sinkhorn normalization for gs_iters iterations (larger -> closer to true permutation) to get a doubly stochastic matrix.
        Args:
            gs_tau: Gumbel-Sinkhorn temperature (lower -> closer to true permutation)
            gs_iters: Number of Sinkhorn iterations (larger -> closer to true permutation)
        """
        super().__init__()
        self.gs_tau = gs_tau
        self.gs_iters = gs_iters

    # ---- Gumbel noise sampling ----
    def sample_gumbel(self, shape: torch.Size, eps: float, device: torch.device) -> torch.Tensor:
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
        gumbel_noise = self.sample_gumbel(scores.shape, 1e-20, scores.device)
        scores = (scores + gumbel_noise) / self.gs_tau
        # Sinkhorn normalization to get doubly stochastic matrix
        soft_perm = self.sinkhorn(scores)
        return soft_perm


class TourConstructor(nn.Module):
    def __init__(self, method: str):
        """
        Creates a hard tour from the given soft permutation matrix.
        Applies straight-through estimation for gradient flow.
        Args:
            method: (str) - method for hard permutation extraction ("greedy" or "hungarian")
        """
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

    def forward(self, soft_perm: torch.Tensor) -> torch.Tensor:
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


class ConvolutionBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int):
        """
        A standard residual block for 2D convolutions, inspired by ResNet.
        Processes an image-like tensor (e.g., a permutation matrix).
        The block structure is: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            kernel_size: Size of the convolutional kernel (assumed square)
            stride: Stride for the first convolutional layer
        """
        super(ConvolutionBlock, self).__init__()
        padding = kernel_size // 2
        self.relu = nn.ReLU()

        # Main Path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, 1, padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Residual Path
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_C, N, N) - A 4D tensor representing a batch of images or feature maps.
        Returns:
            x: (B, out_C, N, N) - The processed tensor.
        """
        # Skip connection
        identity = self.shortcut(x)
        # Conv2D Layer 1
        out = self.conv1(x)
        out = self.bn1(out)
        # ReLU Activation 1
        out = self.relu(out)
        # Conv2D Layer 2
        out = self.conv2(out)
        out = self.bn2(out)
        # Residual connection and ReLU Activation 2
        out = out + identity
        out = self.relu(out)
        return out


class MLP(nn.Module):
    def __init__(self, input_dim: int, feed_forward_dim: int, embedding_dim: int, dropout: float, output_dim: int):
        """
        A simple feedforward neural network with 3 linear layers, ReLU activations, and dropout.
        Designed for regression tasks, outputting a single scalar value.
        Args:
            input_dim: Dimension of the input features
            feed_forward_dim: Dimension of the hidden layers
            embedding_dim: Dimension of the second hidden layer
            output_dim: Dimension of the output (1 for regression)
        """
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, feed_forward_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(feed_forward_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) - A 2D tensor representing a batch of input features.
        Returns:
            out: (B, 1) - A tensor representing the output scalar for each input in the batch.
        """
        return self.network(x)
