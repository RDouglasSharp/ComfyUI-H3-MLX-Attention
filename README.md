# ComfyUI-H3-MLX-Attention

Runs MiniMax H3's DiT self-attention through Apple's MLX (`mx.fast.scaled_dot_product_attention`) instead of ComfyUI's stock MPS attention. Apple Silicon only. There are no nodes to add to a graph: once installed, it applies to every H3 workflow (t2v, i2v, first+last frame, ref2va).

## Why

On MPS, ComfyUI's H3 attention uses sub-quadratic attention (`attention_sub_quad`). It is slower than MLX's fused kernel. At long sequence lengths and under memory pressure it can also return wrong results for some heads, which shows up as dark or broken video (Comfy-Org/ComfyUI#17418). MLX's kernel was correct in every case we checked: worst per-head error against an FP64 reference was 1.7e-3, the BF16 floor.

## Results

Tested on a **14-inch MacBook Pro, M5 Max, 128 GB**, with fans held at maximum, macOS 27.0.1, ComfyUI 0.37.2 (also smoke-tested on 0.44.2), PyTorch 2.12.1, MLX 0.32.4. The model was MiniMax H3 FL2VA INT8 (convrot) with ComfyUI-AppleSilicon-FP8 1.4.0, 20 steps res_multistep. The 14-inch chassis has limited cooling (about 97 W input cap), so absolute times on a 16-inch or a desktop Mac will differ. The cycle counts below are the comparison that transfers between machines.

| Clip | Tokens | GPU cycles per step, stock → MLX | Change |
|---|---|---|---|
| 73 frames, 832×480 | 8,964 | 20.9 → 19.25 G | −7.9% |
| 124 frames (5 s), 832×480 | 14,984 | 38.1 → 34.06 G | −10.6% |
| 141 frames (6 s), 1088×608 | 27,742 | 96.4 → 75.4 G | −21.8% |

- 15 s clip (362 frames, 832×480): 158.9 s/step with MLX vs 206.6 with the stock backend (−23%), and peak RAM 80 GB vs 97–105 GB.
- Same-day heat-soaked wall time per step: 18–27% faster across those lengths.

## Install

1. Clone or copy this folder into `ComfyUI/custom_nodes/`.
2. Install MLX into ComfyUI's Python environment, for example `ComfyUI/.venv/bin/python -m pip install -r custom_nodes/ComfyUI-H3-MLX-Attention/requirements.txt`.
3. Restart ComfyUI. The log shows one line, either `[H3-MLX-Attention] active: ...` or `inactive (<reason>)`.

## Switch it off

Set `"enabled": false` in `settings.json`, or start ComfyUI with `H3_MLX_ATTN=0`. The environment variable wins.

## How it works

- q/k/v are made contiguous and handed to MLX zero-copy via DLPack; the output returns through `torch.from_dlpack`.
- MLX and PyTorch use separate Metal command queues, so each call runs `torch.mps.synchronize()` before MLX reads and `mx.eval()` before returning. That costs about 1.4% of a step in idle GPU time.
- MLX's buffer cache is capped (`cache_gb`, default 2) so it doesn't add to ComfyUI's peak memory.
- Masks, non-bf16 inputs, non-MPS tensors or unexpected shapes fall back to the stock backend. Each fallback reason is logged once.
- `H3_MLX_ATTN_TIMING=1` logs per-call handoff timings.

## Limits

- Only MiniMax H3's DiT attention is routed. Other models are untouched.
- Patching an internal name (`comfy.ldm.minimax.model.optimized_attention`) may break with future ComfyUI changes. In that case it logs a fallback or an inactive line and ComfyUI keeps working.
- Separately from this node, H3 clips above about 34k tokens currently go black on MPS with any attention backend (seen on ComfyUI 0.37.2 and 0.44.2; see Comfy-Org/ComfyUI#15799).

## License

MIT
