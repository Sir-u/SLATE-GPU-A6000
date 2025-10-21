import math
from typing import Optional, Tuple

import torch
from torch import nn

"""
SLATE: Sliding Local Attention with Two-pass Encodings.
Per layer:
    1. Sort nodes (incl. depot index 0) by x, apply sliding-window MH attention.
    2. Sort by y, apply same.
    3. Windows are centered with adaptive boundary shift (no padding). Base size = W.
    4. Depot always appended to each local window (neighbors preserved) -> logical window size W+1.
    5. Distance decay: subtract (Δcoord)^2 / tau from logits (axis-wise squared distance).
    6. Depot gets an extra full attention pass for global mixing (O(T)) per layer.
    7. Axis outputs fused by mean (0.5*(hx + hy)); residual + norm + FFN + residual + norm.
Complexity ≈ O(L * N * W) + O(L * N) vs original O(L * N^2).
Use --spatial_encoder to enable pathway.
"""

# ---------------------------------------------------------------------
# Normalization (reuse pattern from original code: batch or instance)
# ---------------------------------------------------------------------
class _Normalization(nn.Module):
    def __init__(self, embed_dim: int, normalization: str = 'instance'):
        super().__init__()
        mapping = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }
        self.norm_type = normalization
        self.norm = mapping.get(normalization, nn.BatchNorm1d)(embed_dim, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, E)
        if self.norm_type == 'batch':
            return self.norm(x.view(-1, x.size(-1))).view(*x.size())
        elif self.norm_type == 'instance':
            return self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)
        return x

# ---------------------------------------------------------------------
# Axis-specific sliding window attention with distance decay
# ---------------------------------------------------------------------
class _AxisSlidingAttention(nn.Module):
    """Sliding-window multi-head self-attention with mandatory global depot inclusion.
    Every position's window always contains depot (index 0), and depot itself attends full sequence.
    Distance decay subtracts (delta_coord)^2 / tau from scores.
    """
    def __init__(self, embed_dim: int, n_heads: int, window_size: int, distance_tau: float):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.window_size = window_size
        self.tau = distance_tau
        self.Wq = nn.Linear(embed_dim, embed_dim)
        self.Wk = nn.Linear(embed_dim, embed_dim)
        self.Wv = nn.Linear(embed_dim, embed_dim)
        self.Wo = nn.Linear(embed_dim, embed_dim)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def _build_windows(self, T: int, device: torch.device) -> torch.Tensor:
        """Centered sliding window indices (no depot forcing). Returns (T, W_effective)."""
        W = min(self.window_size, T)
        if W <= 0:
            return torch.zeros((T, 0), dtype=torch.long, device=device)
        half = W // 2
        positions = torch.arange(T, device=device)
        start = positions - half
        start.clamp_(min=0)
        end = start + W
        overflow = (end - T).clamp_min(0)
        start = (start - overflow).clamp_min(0)
        rel = torch.arange(W, device=device)  # (W,)
        idx = start.unsqueeze(1) + rel  # (T, W)
        return idx

    def forward(self, h: torch.Tensor, coord_1d: torch.Tensor) -> torch.Tensor:
        B, T, E = h.shape
        if T <= 1:
            return h

        # Sorting for locality
        sort_idx = torch.argsort(coord_1d, dim=1)
        inv_idx = torch.argsort(sort_idx, dim=1)
        h_sorted = torch.gather(h, 1, sort_idx.unsqueeze(-1).expand(-1, -1, E))
        coord_sorted = torch.gather(coord_1d, 1, sort_idx)

        Q = self.Wq(h_sorted).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,T,D)
        K = self.Wk(h_sorted).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.Wv(h_sorted).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        if self.window_size >= T:  # full attention for all
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            diff = coord_sorted.unsqueeze(2) - coord_sorted.unsqueeze(1)
            dist2 = diff.pow(2)
            scores = scores - dist2.unsqueeze(1) / self.tau
            attn = torch.softmax(scores, dim=-1)
            ctx = torch.matmul(attn, V)
            # Apply output projection after concatenating heads
            out_sorted = self.Wo(ctx.transpose(1, 2).reshape(B, T, E))
        else:
            # Local windows (T,W) with per-batch depot forcing
            base_idx = self._build_windows(T, h.device)  # (T,W)
            W = base_idx.size(1)

            # Depot position after sorting for each batch: d[b] in [0..T-1]
            depot_seq_idx = (sort_idx == 0).int().argmax(dim=1)  # (B,)

            # Append depot index per-batch → (B,T,W+1)
            idx_ext = base_idx.unsqueeze(0).expand(B, -1, -1).clone()              # (B,T,W)
            idx_ext = torch.cat([idx_ext, depot_seq_idx.view(B, 1, 1).expand(-1, T, 1)], dim=2)  # (B,T,W+1)

            # If window already contains depot, mask the appended column
            has_depot = (base_idx.unsqueeze(0) == depot_seq_idx.view(B, 1, 1)).any(dim=2)  # (B,T)
            keep_mask = torch.ones(B, T, W + 1, dtype=torch.bool, device=h.device)         # (B,T,W+1)
            keep_mask[:, :, -1] = ~has_depot

            # Gather K,V for local windows (B,H,T,W+1,D) without T×T expansion
            idx5 = idx_ext.unsqueeze(1).unsqueeze(-1).expand(-1, self.n_heads, -1, -1, self.head_dim)
            K_w = torch.gather(K.unsqueeze(3).expand(-1, -1, -1, W + 1, -1), 2, idx5)
            V_w = torch.gather(V.unsqueeze(3).expand(-1, -1, -1, W + 1, -1), 2, idx5)

            # Compute local scores with distance decay
            scores_local = (Q.unsqueeze(3) * K_w).sum(-1) * self.scale  # (B,H,T,W+1)
            # Gather coord windows with matching rank using take_along_dim
            coord_windows = torch.take_along_dim(coord_sorted.unsqueeze(-1), idx_ext, dim=1)  # (B,T,W+1)
            dist2 = (coord_windows - coord_sorted.unsqueeze(-1)).pow(2)   # (B,T,W+1)
            scores_local = scores_local - dist2.unsqueeze(1) / self.tau

            # Stable masking (dtype min avoids NaNs under AMP/mixed precision)
            neg = torch.finfo(scores_local.dtype).min
            if (~keep_mask).any():
                scores_local = scores_local.masked_fill(~keep_mask.unsqueeze(1), neg)

            attn_local = torch.softmax(scores_local, dim=-1)
            ctx_local = (attn_local.unsqueeze(-1) * V_w).sum(-2)  # (B,H,T,D)
            # Apply output projection after concatenating heads
            local_out = self.Wo(ctx_local.transpose(1, 2).reshape(B, T, E))

            # Depot full attention override (vectorized)
            d_idx = depot_seq_idx.view(B, 1, 1, 1).expand(-1, self.n_heads, 1, self.head_dim)
            Qd = torch.gather(Q, 2, d_idx)  # (B,H,1,D)
            scores_d = torch.matmul(Qd, K.transpose(-2, -1)) * self.scale  # (B,H,1,T)
            coord_d = torch.gather(coord_sorted, 1, depot_seq_idx.view(B, 1))  # (B,1)
            dist_d2 = (coord_sorted - coord_d).pow(2)  # (B,T)
            scores_d = scores_d - dist_d2.unsqueeze(1).unsqueeze(2) / self.tau
            attn_d = torch.softmax(scores_d, dim=-1)
            ctx_d = torch.matmul(attn_d, V).transpose(1, 2).reshape(B, 1, E)
            # Output projection for depot context as well
            ctx_d = self.Wo(ctx_d)

            # Replace depot output back to its sorted position (per-batch row replace)
            out_sorted = local_out.clone()
            out_sorted[torch.arange(B, device=h.device), depot_seq_idx] = ctx_d.squeeze(1)

        out = torch.gather(out_sorted, 1, inv_idx.unsqueeze(-1).expand(-1, -1, E))
        return out

# ---------------------------------------------------------------------
# One dual-axis layer: X + Y sliding attention + mean fuse + FFN
# ---------------------------------------------------------------------
class _SpatialDualAxisLayer(nn.Module):
    def __init__(self, embed_dim: int, n_heads: int, window_size: int, distance_tau: float,
                 feed_forward_hidden: int = 512, normalization: str = 'batch'):
        super().__init__()
        self.axis_x = _AxisSlidingAttention(embed_dim, n_heads, window_size, distance_tau)
        self.axis_y = _AxisSlidingAttention(embed_dim, n_heads, window_size, distance_tau)
        self.ff = (nn.Sequential(
            nn.Linear(embed_dim, feed_forward_hidden),
            nn.ReLU(),
            nn.Linear(feed_forward_hidden, embed_dim)
        ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim))
        self.norm_attn = _Normalization(embed_dim, normalization)
        self.norm_ff = _Normalization(embed_dim, normalization)

    def forward(self, h: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        B, N, E = h.shape
        if N <= 1:
            return h
        hx = self.axis_x(h, coords[:, :, 0])
        hy = self.axis_y(h, coords[:, :, 1])
        fused = 0.5 * (hx + hy)
        h_attn = self.norm_attn(h + fused)
        ff_out = self.ff(h_attn)
        h_out = self.norm_ff(h_attn + ff_out)
        return h_out

# ---------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------
class SpatialDualAxisEncoder(nn.Module):
    """Encoder replacement with mandatory dual-axis sliding attention + distance decay.
    Parameters
    ----------
    n_heads : int
    embed_dim : int
    n_layers : int
    node_dim : int (if None, assumes input already in embed_dim)
    normalization : 'batch' | 'instance'
    feed_forward_hidden : hidden size for FFN (<=0 -> linear)
    window_size : sliding window size (default 32)
    distance_tau : decay temperature (default 0.15 for coords in [0,1])
    """
    DEFAULT_WINDOW_SIZE = 32
    DEFAULT_DISTANCE_TAU = 0.15

    def __init__(self,
                 n_heads: int,
                 embed_dim: int,
                 n_layers: int,
                 node_dim: Optional[int] = None,
                 normalization: str = 'instance',
                 feed_forward_hidden: int = 512,
                 window_size: Optional[int] = None,
                 distance_tau: Optional[float] = None):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None
        w = window_size if window_size is not None else self.DEFAULT_WINDOW_SIZE
        tau = distance_tau if distance_tau is not None else self.DEFAULT_DISTANCE_TAU
        self.layers = nn.ModuleList([
            _SpatialDualAxisLayer(embed_dim, n_heads, w, tau, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        assert mask is None, "mask not supported in spatial encoder"
        h = (self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1)
             if self.init_embed is not None else x)
        coords = x[..., :2]
        for layer in self.layers:
            h = layer(h, coords)
        graph_embed = h.mean(dim=1)
        return h, graph_embed

__all__ = [
    'SpatialDualAxisEncoder'
]
