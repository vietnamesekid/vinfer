# Qwen3.5 (hybrid Gated DeltaNet + Gated Attention): Qwen3.5-2B

Sources read: `config.json`, `model.safetensors.index.json` and the safetensors header of
`Qwen/Qwen3.5-2B` at the revision pinned in `bench/models.lock` (`15852e8c`), and
`transformers/models/qwen3_5/modeling_qwen3_5.py` plus `cache_utils.py` (transformers 5.22.2).
The checkpoint is `Qwen3_5ForConditionalGeneration` (text + vision); the engine uses the text
model only and skips `model.visual.*` and `mtp.*`.

## Config (`text_config`)

| Field | Value |
|---|---|
| `hidden_size` | 2048 |
| `num_hidden_layers` | 24; `layer_types`: 3 × `linear_attention` then 1 × `full_attention`, six times (full attention at layers 3, 7, …, 23) |
| `intermediate_size` | 6144 (same SwiGLU MLP after every token mixer) |
| `vocab_size` | 248,320 |
| `rms_norm_eps` | 1e-6 |
| `tie_word_embeddings` | true (no `lm_head.weight` in the checkpoint) |
| Gated Attention | `num_attention_heads` 8, `num_key_value_heads` 2, `head_dim` 256, `attn_output_gate` true |
| RoPE | `rope_theta` 1e7, `partial_rotary_factor` 0.25 (64 of 256 dims), `mrope_section` [11, 11, 10], `mrope_interleaved` true (all in `rope_parameters`) |
| Gated DeltaNet | `linear_num_key_heads` 16, `linear_num_value_heads` 16, `linear_key_head_dim` 128, `linear_value_head_dim` 128, `linear_conv_kernel_dim` 4 |
| `mamba_ssm_dtype` | float32 |
| MTP | `mtp_num_hidden_layers` 1 (weights under `mtp.*`; not run by HF Transformers) |

Note: in this checkpoint the number of value heads equals the number of key heads and the key
and value head dims are equal. Two classes of bug therefore stay invisible on it (GQA inside the
DeltaNet, and a transposed recurrent state); unit tests use tiny configs where they differ.

## Layer structure

Every layer: `x = x + mixer(input_layernorm(x))`, then `x = x + mlp(post_attention_layernorm(x))`.
Final `norm`, then `lm_head` (tied to `embed_tokens`).

### RMSNorm, three variants

| Where | Formula | Weight init |
|---|---|---|
| `input_layernorm`, `post_attention_layernorm`, final `norm`, attention `q_norm`/`k_norm` (`Qwen3_5RMSNorm`) | `(x_f32 * rsqrt(mean(x_f32²) + eps) * (1 + w_f32)).to(dtype)` — multiply in fp32, then cast | zeros |
| DeltaNet output norm (`Qwen3_5RMSNormGated`), per value head over `head_v_dim` | `y = w * (x_f32 * rsqrt(mean(x_f32²) + eps)).to(dtype)`; `out = (y * silu(z_f32)).to(dtype)` | ones |
| Qwen3 (for comparison) | `w * normed.to(dtype)` | ones |

### Gated Attention (`full_attention` layers)

```
q_and_gate = q_proj(x).view(..., heads, 2 * head_dim)   # per head: [q_h | gate_h]
q, gate = chunk(q_and_gate, 2, dim=-1); gate = gate.reshape(..., heads * head_dim)
q = q_norm(q); k = k_norm(k_proj(x)); v = v_proj(x)      # norms per head, before RoPE
q, k = rope(q, k)                                        # first 64 dims only, NeoX layout
o = attention(q, k, v)                                   # causal, scale head_dim^-0.5, GQA
out = o_proj(o * sigmoid(gate))
```

RoPE: `inv_freq = 1 / theta^(arange(0, 64, 2) / 64)`. With text only, the three MRoPE position
rows are identical, so the interleaved recomposition returns the plain 1-D frequencies: the result
is standard NeoX RoPE on the first 64 dims of each head, the other 192 dims pass through.

### Gated DeltaNet (`linear_attention` layers)

```
qkv = in_proj_qkv(x)                       # [key_dim, key_dim, value_dim] = 2048 + 2048 + 2048
qkv = silu(causal_conv1d(qkv))             # depthwise, kernel 4, no bias, over the time axis
q, k, v = split(qkv); reshape to heads     # q, k: [Hk, Dk]; v: [Hv, Dv]
z = in_proj_z(x)                           # [Hv, Dv], output gate
beta = sigmoid(in_proj_b(x))               # [Hv]
g = -exp(A_log_f32) * softplus(in_proj_a(x)_f32 + dt_bias)   # [Hv], log decay, <= 0
q, k repeated to Hv heads (v-head j uses k-head j // (Hv / Hk))
q, k = l2norm(q), l2norm(k)                # fp32, x * rsqrt(sum(x²) + 1e-6)
q = q * Dk^-0.5
recurrence per value head (state S: [Dk, Dv] in HF, fp32):
    S = S * exp(g_t)
    S = S + k_t ⊗ (beta_t * (v_t - S^T k_t))
    o_t = S^T q_t
out = out_proj(gated_rmsnorm(o, z))
```

- HF casts q, k, v, beta, g to fp32 before the recurrence, so the recurrent state is fp32 in a
  forward pass and in HF's cache (`zeros_like` of the fp32 state). This matches
  `mamba_ssm_dtype: float32`.
- Prefill uses the chunked form (chunk 64, UT transform with `solve_triangular`); single-token
  decode uses the recurrent form. Both compute the same recurrence.
- Conv state: the last `kernel - 1 = 3` **raw** inputs to the conv (`qkv` before conv and SiLU),
  per channel.
- Padding: HF multiplies the layer input by the 2-D attention mask before the DeltaNet, so padded
  positions enter with zero input.

## Tensor names and dtypes (checkpoint)

```
model.language_model.embed_tokens.weight                     BF16 [248320, 2048]
model.language_model.layers.{i}.input_layernorm.weight       BF16 [2048]
model.language_model.layers.{i}.post_attention_layernorm.weight BF16 [2048]
model.language_model.layers.{i}.mlp.{gate,up}_proj.weight    BF16 [6144, 2048]
model.language_model.layers.{i}.mlp.down_proj.weight         BF16 [2048, 6144]
# linear_attention layers
model.language_model.layers.{i}.linear_attn.in_proj_qkv.weight BF16 [6144, 2048]
model.language_model.layers.{i}.linear_attn.in_proj_z.weight   BF16 [2048, 2048]
model.language_model.layers.{i}.linear_attn.in_proj_{a,b}.weight BF16 [16, 2048]
model.language_model.layers.{i}.linear_attn.conv1d.weight      BF16 [6144, 1, 4]
model.language_model.layers.{i}.linear_attn.dt_bias            BF16 [16]
model.language_model.layers.{i}.linear_attn.A_log              F32  [16]
model.language_model.layers.{i}.linear_attn.norm.weight        F32  [128]
model.language_model.layers.{i}.linear_attn.out_proj.weight    BF16 [2048, 2048]
# full_attention layers
model.language_model.layers.{i}.self_attn.q_proj.weight      BF16 [4096, 2048]   # 8 heads × (256 q + 256 gate)
model.language_model.layers.{i}.self_attn.{k,v}_proj.weight  BF16 [512, 2048]
model.language_model.layers.{i}.self_attn.o_proj.weight      BF16 [2048, 2048]
model.language_model.layers.{i}.self_attn.{q,k}_norm.weight  BF16 [256]
model.language_model.norm.weight                             BF16 [2048]
```

`A_log` and the DeltaNet `norm.weight` are stored in fp32. HF Transformers declares no
fp32-kept modules for this model, and loading it with `dtype=bfloat16` rounds both to bf16
(checked with transformers 5.22.2 on a tiny checkpoint laid out like this one). The engine keeps
the stored dtype, so this is one known source of difference from HF in bf16 parity runs.
