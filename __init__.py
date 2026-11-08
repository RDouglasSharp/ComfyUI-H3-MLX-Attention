"""ComfyUI-H3-MLX-Attention: run MiniMax H3's DiT self-attention through MLX on Apple Silicon.

A startup patch with no graph nodes. It wraps `optimized_attention` inside
comfy.ldm.minimax.model (the only attention call site in the H3 DiT and its token refiner) so
that q/k/v, which arrive as [1, heads, S, 128] bf16 MPS tensors, are evaluated with
mx.fast.scaled_dot_product_attention (BF16 operands, FP32 softmax statistics). The result is
returned as an MPS tensor shaped [B, S, heads*128], the same contract as the stock backend.

Handoff: q/k/v are made contiguous, passed to MLX zero-copy via DLPack, and the output comes
back through torch.from_dlpack. MLX and PyTorch use separate Metal command queues, so the
patch calls torch.mps.synchronize() before MLX reads and mx.eval() before returning.

Memory: q/k/v stay owned by PyTorch; MLX's allocator cache is capped (settings.json cache_gb)
so freed MLX buffers are not kept next to ComfyUI's resident peak.

Switches: settings.json {"enabled": false} or the environment variable H3_MLX_ATTN=0 (the
environment wins). H3_MLX_ATTN_TIMING=1 logs per-call handoff timings every 50 calls. Any
error or unexpected input (mask, non-bf16, non-MPS, other shapes) falls back to the stock
backend; each distinct reason is logged once. The startup log states whether it is active.
"""

import json
import os

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}


def _load_settings():
    settings = {"enabled": True, "cache_gb": 2.0}
    try:
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "settings.json")) as f:
            settings.update(json.load(f))
    except Exception:
        pass
    return settings


def _resolve_reason(settings):
    env = os.environ.get("H3_MLX_ATTN")
    if env is not None:
        on = env.strip().lower() not in ("", "0", "false", "no", "off")
        return on, f"H3_MLX_ATTN={env}"
    return bool(settings["enabled"]), f"settings.json enabled={settings['enabled']}"


_SETTINGS = _load_settings()
ARMED, _ARMED_REASON = _resolve_reason(_SETTINGS)

if ARMED:
    try:
        import mlx.core as mx
        import torch

        import comfy.ldm.minimax.model as h3_model
        from comfy.ldm.modules.attention import AttentionTensorContainer
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
    except Exception as e:
        ARMED, _ARMED_REASON = False, f"unavailable: {e!r}"

if ARMED:
    import time

    CACHE_LIMIT = int(float(_SETTINGS.get("cache_gb", 2.0)) * 1024**3)
    mx.metal.set_cache_limit(CACHE_LIMIT)

    _orig_optimized_attention = h3_model.optimized_attention
    _fallbacks = {}
    _timing = os.environ.get("H3_MLX_ATTN_TIMING") == "1"
    _timing_stats = {"n": 0, "sync": 0.0, "copy": 0.0, "mlx": 0.0, "ret": 0.0}

    def _log_fallback(reason):
        if reason not in _fallbacks:
            _fallbacks[reason] = True
            print(f"[H3-MLX-Attention] falling back to the stock backend: {reason}",
                  flush=True)

    def _mlx_optimized_attention(q, k, v, heads, mask=None, skip_reshape=False,
                                 transformer_options={}, **kwargs):
        try:
            if mask is not None:
                raise ValueError("mask not supported")
            if not skip_reshape:
                raise ValueError("expected skip_reshape=True layout")
            if not all(isinstance(t, AttentionTensorContainer) for t in (q, k, v)):
                raise ValueError("q/k/v are not attention tensor containers")
            qt, kt, vt = q.peek(), k.peek(), v.peek()
            if qt.device.type != "mps" or qt.dtype != torch.bfloat16:
                raise ValueError(f"dtype/device {qt.dtype}/{qt.device} not routed")
            if qt.ndim != 4 or qt.shape[1] != heads or qt.shape[-1] != kt.shape[-1]:
                raise ValueError(f"unexpected shape {tuple(qt.shape)} for heads={heads}")
            t0 = time.perf_counter() if _timing else 0.0
            qc, kc, vc = qt.contiguous(), kt.contiguous(), vt.contiguous()
            t1 = time.perf_counter() if _timing else 0.0
            # separate Metal command queues: drain torch's queue before MLX reads
            torch.mps.synchronize()
            t2 = time.perf_counter() if _timing else 0.0
            mq, mk, mv = mx.from_dlpack(qc), mx.from_dlpack(kc), mx.from_dlpack(vc)
            out = mx.fast.scaled_dot_product_attention(
                mq, mk, mv, scale=qt.shape[-1] ** -0.5)
            mx.eval(out)
            t3 = time.perf_counter() if _timing else 0.0
            o = torch.from_dlpack(out)
            if _timing:
                t4 = time.perf_counter()
                st = _timing_stats
                st["n"] += 1
                st["copy"] += t1 - t0
                st["sync"] += t2 - t1
                st["mlx"] += t3 - t2
                st["ret"] += t4 - t3
                if st["n"] % 50 == 0:
                    n = st["n"]
                    print(f"[H3-MLX-Attention][timing] calls 1-{n} (last S={qc.shape[2]}): "
                          f"copy {st['copy'] / n * 1000:.2f} ms/call, "
                          f"sync {st['sync'] / n * 1000:.2f} ms/call, "
                          f"mlx {st['mlx'] / n * 1000:.2f} ms/call, "
                          f"ret {st['ret'] / n * 1000:.2f} ms/call", flush=True)
            return o.transpose(1, 2).flatten(start_dim=2)
        except Exception as e:
            _log_fallback(repr(e))
            return _orig_optimized_attention(q, k, v, heads, mask=mask,
                                             skip_reshape=skip_reshape,
                                             transformer_options=transformer_options,
                                             **kwargs)

    h3_model.optimized_attention = _mlx_optimized_attention
    print(f"[H3-MLX-Attention] active: H3 DiT attention -> mx.fast.scaled_dot_product_attention "
          f"({_ARMED_REASON}, cache limit {CACHE_LIMIT // 1024**2} MB)", flush=True)
else:
    print(f"[H3-MLX-Attention] inactive: H3 DiT attention stays on the stock backend "
          f"({_ARMED_REASON})", flush=True)
