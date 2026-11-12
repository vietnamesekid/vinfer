# Qwen3 (dense, standard attention): Qwen3-0.6B

The engine's first model: standard attention only, so forward pass, loader and sampler bugs can
be separated from Gated DeltaNet bugs. Same tokenizer family as Qwen3.5.

Sources read: `config.json` of `Qwen/Qwen3-0.6B` at the revision pinned in `bench/models.lock`
(`c1899de2`), and `transformers/models/qwen3/modeling_qwen3.py` (transformers 5.22.2).

## Config

| Field | Value |
|---|---|
| `hidden_size` | 1024 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` / `num_key_value_heads` | 16 / 8 (GQA, group 2) |
| `head_dim` | 128 (so `q_proj` is 1024 → 2048, wider than `hidden_size`) |
| `intermediate_size` | 3072 |
| `vocab_size` | 151,936 |
| `rms_norm_eps` | 1e-6 |
| `rope_theta` | 1,000,000; full rotary over `head_dim`; no scaling |
| `attention_bias` | false |
| `tie_word_embeddings` | true; the checkpoint still stores `lm_head.weight` (see below) |
| `sliding_window` | none |
| `torch_dtype` | bfloat16 |

## Forward pass

```
x = embed_tokens[ids]
for each of 28 layers:
    h = x + o_proj(attention(q_norm(q), k_norm(k), v))     # input_layernorm first
    x = h + down_proj(silu(gate_proj(n)) * up_proj(n))     # n = post_attention_layernorm(h)
logits = lm_head(norm(x))                                  # lm_head = embed_tokens (tied)
```

- **RMSNorm** (`Qwen3RMSNorm`): `weight * (x_f32 * rsqrt(mean(x_f32²) + eps)).to(input_dtype)`.
  The cast back to the input dtype happens **before** multiplying by the weight. Weight
  initialized to ones.
- **q/k norm**: RMSNorm over `head_dim` per head, applied before RoPE.
- **RoPE**: NeoX layout (`rotate_half` on two halves, `cos`/`sin` = `cat(freqs, freqs)`),
  `inv_freq = 1 / theta^(arange(0, head_dim, 2) / head_dim)`, computed in fp32, cast to the model
  dtype before use.
- **Attention**: causal, scale `head_dim^-0.5`, KV heads repeated for GQA.
- **MLP**: SwiGLU, no bias.

## Tensor names (checkpoint)

```
model.embed_tokens.weight                    [151936, 1024]
model.layers.{i}.input_layernorm.weight      [1024]
model.layers.{i}.self_attn.q_proj.weight     [2048, 1024]
model.layers.{i}.self_attn.k_proj.weight     [1024, 1024]
model.layers.{i}.self_attn.v_proj.weight     [1024, 1024]
model.layers.{i}.self_attn.o_proj.weight     [1024, 2048]
model.layers.{i}.self_attn.q_norm.weight     [128]
model.layers.{i}.self_attn.k_norm.weight     [128]
model.layers.{i}.post_attention_layernorm.weight [1024]
model.layers.{i}.mlp.{gate,up}_proj.weight   [3072, 1024]
model.layers.{i}.mlp.down_proj.weight        [1024, 3072]
model.norm.weight                            [1024]
lm_head.weight                               [151936, 1024]
```

All BF16; 311 tensors (read from the safetensors header). With tied embeddings HF uses
`embed_tokens` as the output projection; the engine does the same and, when `lm_head.weight` is
present, checks that it equals `embed_tokens.weight` instead of silently picking one.
