"""Readable square self-attention baselines; no dropout, padding or GQA."""
import math
import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.autograd.function import once_differentiable


def validate(q, k, v):
    if q.ndim != 4 or q.shape != k.shape:
        raise ValueError('Q and K must have matching [batch, heads, tokens, dim] shapes')
    if v.ndim != 4 or v.shape[:3] != q.shape[:3]:
        raise ValueError('V must match Q batch, heads and tokens')
    if any(size == 0 for size in (*q.shape, v.shape[-1])):
        raise ValueError('All dimensions must be nonzero')
    if not (q.dtype == k.dtype == v.dtype and q.device == k.device == v.device):
        raise ValueError('Inputs must share dtype and device')
    if not all(x.is_floating_point() for x in (q, k, v)):
        raise ValueError('Inputs must be floating point')


def reference(q, k, v, *, causal=False):
    validate(q, k, v)
    scores = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        n = q.shape[-2]
        allowed = torch.ones(n, n, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~allowed, -torch.inf)
    return torch.softmax(scores, dim=-1) @ v


def chunked(q, k, v, *, causal=False, chunk_size=128):
    """Chunk queries, not keys: every query still attends to all permitted keys.

    In inference this bounds each score matrix to chunk_size × N. In training,
    ordinary autograd may retain intermediates from every chunk. Not FlashAttention.
    """
    validate(q, k, v)
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    n, dim = q.shape[-2:]
    outputs = []
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        scores = (q[..., start:stop, :] @ k.transpose(-2, -1)) / math.sqrt(dim)
        if causal:
            allowed = torch.arange(n, device=q.device)[None, :] <= torch.arange(
                start, stop, device=q.device
            )[:, None]
            scores = scores.masked_fill(~allowed, -torch.inf)
        outputs.append(torch.softmax(scores, dim=-1) @ v)
    return torch.cat(outputs, dim=-2)


def pytorch_math(q, k, v, *, causal=False):
    validate(q, k, v)
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=0., is_causal=causal)


class _RecomputeAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, chunk_size):
        n, dim = q.shape[-2:]
        outputs, normalizers = [], []
        for start in range(0, n, chunk_size):
            stop = min(start + chunk_size, n)
            scores = (q[..., start:stop, :] @ k.transpose(-2, -1)) / math.sqrt(dim)
            if causal:
                allowed = torch.arange(n, device=q.device)[None, :] <= torch.arange(start, stop, device=q.device)[:, None]
                scores = scores.masked_fill(~allowed, -torch.inf)
            lse = torch.logsumexp(scores, dim=-1, keepdim=True)
            probabilities = torch.exp(scores - lse)
            outputs.append(probabilities @ v)
            normalizers.append(lse)
        ctx.save_for_backward(q, k, v, torch.cat(normalizers, dim=-2))
        ctx.causal, ctx.chunk_size = causal, chunk_size
        return torch.cat(outputs, dim=-2)

    @staticmethod
    @once_differentiable
    def backward(ctx, upstream):
        # Analytic first-order backward. No nested autograd graph or N×N cache.
        q, k, v, lse = ctx.saved_tensors
        n, dim = q.shape[-2:]
        scale = 1 / math.sqrt(dim)
        dq, dk, dv = torch.zeros_like(q), torch.zeros_like(k), torch.zeros_like(v)
        for start in range(0, n, ctx.chunk_size):
            stop = min(start + ctx.chunk_size, n)
            qi, gi = q[..., start:stop, :], upstream[..., start:stop, :]
            scores = (qi @ k.transpose(-2, -1)) * scale
            if ctx.causal:
                allowed = torch.arange(n, device=q.device)[None, :] <= torch.arange(start, stop, device=q.device)[:, None]
                scores = scores.masked_fill(~allowed, -torch.inf)
            p = torch.exp(scores - lse[..., start:stop, :])
            dp = gi @ v.transpose(-2, -1)
            ds = p * (dp - (p * dp).sum(dim=-1, keepdim=True))
            dq[..., start:stop, :] = (ds @ k) * scale
            dk.add_((ds.transpose(-2, -1) @ qi) * scale)
            dv.add_(p.transpose(-2, -1) @ gi)
        return dq, dk, dv, None, None


def recompute(q, k, v, *, causal=False, chunk_size=128):
    """Educational first-order backward; float32/64 only, no dropout or double backward."""
    validate(q, k, v)
    if chunk_size <= 0:
        raise ValueError('chunk_size must be positive')
    if q.dtype not in (torch.float32, torch.float64):
        raise ValueError('This baseline is validated only for float32 and float64')
    return _RecomputeAttention.apply(q, k, v, causal, chunk_size)


METHODS = {'reference': reference, 'query_chunked': chunked, 'recomputed_backward': recompute, 'pytorch_math': pytorch_math}
