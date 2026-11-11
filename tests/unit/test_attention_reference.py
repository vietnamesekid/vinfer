import pytest
import torch
import torch.nn.functional as F

from vinfer.kernels.attention import attention_reference


@pytest.mark.parametrize("num_q_heads,num_kv_heads", [(4, 4), (8, 2)])
def test_matches_sdpa_prefill(num_q_heads, num_kv_heads):
    b, s, d = 2, 7, 16
    q = torch.randn(b, num_q_heads, s, d)
    k = torch.randn(b, num_kv_heads, s, d)
    v = torch.randn(b, num_kv_heads, s, d)

    out = attention_reference(q, k, v, causal=True)

    group = num_q_heads // num_kv_heads
    expected = F.scaled_dot_product_attention(
        q, k.repeat_interleave(group, 1), v.repeat_interleave(group, 1), is_causal=True
    )
    torch.testing.assert_close(out, expected, atol=1e-5, rtol=1e-5)


def test_decode_step_sees_whole_cache():
    # A single query token must attend to every cached key, not just the first.
    q = torch.randn(1, 2, 1, 8)
    k = torch.randn(1, 2, 5, 8)
    v = torch.randn(1, 2, 5, 8)
    causal = attention_reference(q, k, v, causal=True)
    full = attention_reference(q, k, v, causal=False)
    torch.testing.assert_close(causal, full)


def test_rejects_bad_gqa_ratio():
    with pytest.raises(ValueError):
        attention_reference(
            torch.randn(1, 3, 1, 8), torch.randn(1, 2, 1, 8), torch.randn(1, 2, 1, 8)
        )
