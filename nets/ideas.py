import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# Implementation of Embedding Net for variable N

class EmbeddingNet(nn.Module):
    """
    Node feature encoder + cyclic positional encoding that supports variable N per batch.
    - Exact periodicity on each instance's tour length N_b.
    - Integer harmonics for stationarity on the circle.
    - Fixed K at init; for each instance, use K_eff = min(K, floor(N_b/2)) to respect Nyquist,
      then zero-pad to 2K so the learned projection has a fixed input width.
    """
    def __init__(self, node_dim: int, embedding_dim: int, k: int, alpha: float, device):
        """
        Args:
            node_dim: input node feature dim
            embedding_dim: model embedding dim for each of (node,pos)
            device: torch device
            k: number of integer harmonics (total pos-chan = 2k before projection)
            alpha: amplitude decay exponent (a_h = h^{-alpha}); 0 => flat
        """
        super().__init__()
        self.node_feature_encoder = nn.Linear(node_dim, embedding_dim, bias=False)  # Linear layer for node feature embedding
        self.cyclic_projection = nn.Linear(2 * k, embedding_dim, bias=False)  # Linear layer to project harmonics to embedding_dim
        # Store parameters
        self.k = k
        self.alpha = alpha
        self.device = device
        # Initialize the cache for cyclic encodings
        self._cache = {}  # key: (N, K_eff, device) -> tensor [N, 2*K_eff]


    @torch.no_grad()
    def cyclic_encoding(self, N: int, K_eff: int, device) -> torch.Tensor:
        """
        Build the active (unpadded) cyclic features for a tour of length N with K_eff harmonics.
        Returns [N, 2*K_eff] tensor on 'device'. No projection, no padding.
        """
        key = (int(N), int(K_eff), device)
        if key in self._cache:
            return self._cache[key]

        t = torch.arange(N, device=device, dtype=torch.float32)  # tour phases: positions 0...N-1 [N]
        h = torch.arange(1, K_eff + 1, device=device, dtype=torch.float32)  # integer harmonics for exact periodicity [K_eff]
        angles = 2 * math.pi * (t[:, None] * h[None, :] / N)  # [N, K_eff]
        feat = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1).reshape(N, 2 * K_eff)  # [N, K_eff, 2] -> [N, 2*K_eff]
        a = h.pow(-self.alpha)  # apply amplitudes per harmonic to both sin and cos channels [K_eff]
        feat = feat * a.repeat_interleave(2).unsqueeze(0)  # [N, 2*K_eff]

        self._cache[key] = feat
        return feat

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        Args:
            x:        [B, N_max, node_dim] Batch features (padded to N_max if variable lengths)
            lengths:  [B] Tour lengths in the batch
        Returns:
            feats: [B, N_max, 2*embedding_dim]  (NFE + CE)
            mask:  [B, N_max] boolean (True for valid positions)
        """
        device = x.device
        B, N_max, _ = x.shape

        # Node Feature Embedding
        x_norm = F.layer_norm(x, (x.shape[-1],))  # [B, N_max, node_dim]
        nfe = self.node_feature_encoder(x_norm)  # [B, N_max, d]

        # Cyclic Embedding
        ce_raw = torch.zeros(B, N_max, 2 * self.k, device=device, dtype=torch.float32)  # tensor for raw cyclic embeddings [B, N_max, 2K]
        ce_mask = torch.zeros(B, N_max, device=device, dtype=torch.bool)  # tensor for cyclic embeddings mask [B, N_max]
        ## Iterate over batch and generate cyclic features
        for b in range(B):
            N_b = int(lengths[b].item())
            k_eff = max(1, min(self.k, N_b // 2))  # Nyquist-safe initialization of harmonics, at least 1 (irrelevant for large N)
            active = self.cyclic_encoding(N_b, k_eff, device)  # [N_b, 2*K_eff]
            active = active * math.sqrt(self.k / max(1, k_eff))  # scale so expected norm is comparable across differing K_eff
            # place active block and pad right with zeros to 2K
            ce_raw[b, :N_b, :2 * k_eff] = active
            ce_mask[b, :N_b] = True
        ## project raw cyclic features to embedding dimension
        ce = self.cyclic_projection(ce_raw)                        # [B, N_max, d]

        # Concatenate node feature embedding and cyclic embedding
        feats = torch.cat([nfe, ce], dim=-1)          # [B, N_max, 2d]
        return feats, ce_mask







# Cyclic Positional Encoding as own module for large N


class CyclicPositional(nn.Module):
    def __init__(self, d: int, K: int = 64, alpha: float = 0.0):
        super().__init__()
        self.d = d
        self.K = K
        self.alpha = alpha
        self.proj = nn.Linear(2*K, d, bias=False)

    @torch.no_grad()
    def _angles_step_params(self, lengths: torch.Tensor):
        # lengths: [B]
        device = lengths.device
        B = lengths.numel()
        k = torch.arange(1, self.K+1, device=device, dtype=torch.float32)  # [K]
        # Δ per sample/harmonic
        Delta = 2*math.pi * k[None, :] / lengths[:, None].float()         # [B,K]
        c = torch.cos(Delta)                                              # [B,K]
        s = torch.sin(Delta)                                              # [B,K]
        a = k.pow(-self.alpha)[None, :]                                   # [1,K]
        return c, s, a

    def forward(self, lengths: torch.Tensor):
        """
        Returns:
          pos_emb: [B, N_max, d]
          mask:    [B, N_max] (True=valid)
        Uses recurrence to avoid computing sin/cos for all t,k explicitly.
        """
        device = lengths.device
        B = lengths.numel()
        N_max = int(lengths.max().item())
        c, s, a = self._angles_step_params(lengths)                       # [B,K] each

        # init phase at t=0: cos=1, sin=0 for every harmonic
        cos_t = torch.ones(B, self.K, device=device)
        sin_t = torch.zeros(B, self.K, device=device)

        # container for [B,N_max,2K]
        feats = torch.zeros(B, N_max, 2*self.K, device=device)

        # scale to keep variance comparable across different K_eff when some N are short
        scale = 1.0 / math.sqrt(self.K)

        for t in range(N_max):
            # write current features: [B,2K]
            cur = torch.stack([cos_t, sin_t], dim=-1).reshape(B, 2*self.K)  # [B,2K]
            cur = cur * a.repeat_interleave(2, dim=1) * scale               # amplitude + variance control
            feats[:, t, :] = cur

            # advance phases once (recurrence): θ_{t+1} = θ_t + Δ
            # cos(θ+Δ)=cosθ·c - sinθ·s;  sin(θ+Δ)=sinθ·c + cosθ·s
            new_cos = cos_t*c - sin_t*s
            new_sin = sin_t*c + cos_t*s
            cos_t, sin_t = new_cos, new_sin

        # mask
        idx = torch.arange(N_max, device=device)[None, :].expand(B, -1)
        mask = idx < lengths[:, None]

        # zero-out padded rows (optional)
        feats[~mask] = 0.0

        # project to model dim
        pos_emb = self.proj(feats)                                         # [B,N_max,d]
        return pos_emb, mask


# Use this if you process one token at a time with Mamba
class CyclicPositionalStreamer(nn.Module):
    def __init__(self, d: int, K: int = 64, alpha: float = 0.0):
        super().__init__()
        self.d, self.K, self.alpha = d, K, alpha
        self.proj = nn.Linear(2*K, d, bias=False)
        self.register_buffer("k_idx", torch.arange(1, K+1).float(), persistent=False)

    def init_state(self, lengths: torch.Tensor):
        # lengths: [B]
        device = lengths.device
        k = self.k_idx.to(device)                     # [K]
        Delta = 2*math.pi * k[None, :] / lengths[:, None].float()  # [B,K]
        c = torch.cos(Delta); s = torch.sin(Delta)    # [B,K]
        a = k.pow(-self.alpha)[None, :].to(device)    # [B,K]
        cos_t = torch.ones_like(c)                    # [B,K]
        sin_t = torch.zeros_like(s)                   # [B,K]
        scale = 1.0 / math.sqrt(self.K)
        return {"c": c, "s": s, "a": a, "cos": cos_t, "sin": sin_t, "scale": scale}

    def step(self, state):
        # build current features
        B, K = state["cos"].shape
        cur = torch.stack([state["cos"], state["sin"]], dim=-1).reshape(B, 2*K)  # [B,2K]
        cur = cur * state["a"].repeat_interleave(2, dim=1) * state["scale"]      # [B,2K]
        pos = self.proj(cur)                                                     # [B,d]

        # advance phases
        cos, sin, c, s = state["cos"], state["sin"], state["c"], state["s"]
        new_cos = cos*c - sin*s
        new_sin = sin*c + cos*s
        state["cos"], state["sin"] = new_cos, new_sin
        return pos, state
