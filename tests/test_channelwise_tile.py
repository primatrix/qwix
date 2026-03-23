"""Tests for channelwise_tile_size (2D block-wise quantization)."""

import jax
import jax.numpy as jnp
import numpy as np
import unittest
from qwix._src.core import dot_general_qt
from qwix._src.core import qarray


class ChannelwiseTileTest(unittest.TestCase):
    """Test that channelwise_tile_size produces correct block-wise scales."""

    def test_channelwise_tile_scale_shape(self):
        """With channelwise_tile_size=128, weight [512, 512] should produce scale [4, 4]."""
        x = jax.random.normal(jax.random.PRNGKey(0), (512, 512), dtype=jnp.bfloat16)
        how = qarray.HowToQuantize(
            qtype=jnp.float8_e4m3fn,
            channelwise_axes=(1,),
            tiled_axes={0: 128},
            channelwise_tile_size=128,
            calibration_method='absmax',
        )
        qa = qarray.quantize(x, how)
        self.assertEqual(qa.scale.shape, (4, 4))

    def test_channelwise_tile_none_keeps_per_element(self):
        """Without channelwise_tile_size, channelwise axis keeps full size in scale."""
        x = jax.random.normal(jax.random.PRNGKey(0), (512, 512), dtype=jnp.bfloat16)
        how = qarray.HowToQuantize(
            qtype=jnp.float8_e4m3fn,
            channelwise_axes=(1,),
            tiled_axes={0: 128},
            channelwise_tile_size=None,
            calibration_method='absmax',
        )
        qa = qarray.quantize(x, how)
        self.assertEqual(qa.scale.shape, (4, 512))

    def test_activation_1x128_tiling(self):
        """Activation [B=32, M=512] with tile_size=128, no channelwise_tile -> scale [32, 4]."""
        x = jax.random.normal(jax.random.PRNGKey(0), (32, 512), dtype=jnp.bfloat16)
        how = qarray.HowToQuantize(
            qtype=jnp.float8_e4m3fn,
            channelwise_axes=(0,),
            tiled_axes={1: 128},
            channelwise_tile_size=None,
            calibration_method='absmax',
        )
        qa = qarray.quantize(x, how)
        self.assertEqual(qa.scale.shape, (32, 4))

    def test_roundtrip_precision(self):
        """Block-wise FP8 roundtrip should have lower error than per-tensor FP8."""
        key1, key2 = jax.random.split(jax.random.PRNGKey(42))
        # Create tensor with blocks of very different magnitudes.
        # 4 blocks along axis 0, each with different scale.
        # Per-tensor scale is dominated by the largest block.
        x = jnp.concatenate([
            jax.random.normal(key1, (128, 128), dtype=jnp.bfloat16) * 1000,
            jax.random.normal(key2, (128, 128), dtype=jnp.bfloat16) * 10,
            jax.random.normal(key1, (128, 128), dtype=jnp.bfloat16) * 0.1,
            jax.random.normal(key2, (128, 128), dtype=jnp.bfloat16) * 0.001,
        ], axis=0)

        how_block = qarray.HowToQuantize(
            qtype=jnp.float8_e4m3fn,
            channelwise_axes=(1,),
            tiled_axes={0: 128},
            channelwise_tile_size=128,
            calibration_method='absmax',
        )
        qa_block = qarray.quantize(x, how_block)
        x_block = qarray.dequantize(qa_block)

        how_tensor = qarray.HowToQuantize(
            qtype=jnp.float8_e4m3fn,
            channelwise_axes=(),
            tiled_axes={},
            channelwise_tile_size=None,
            calibration_method='absmax',
        )
        qa_tensor = qarray.quantize(x, how_tensor)
        x_tensor = qarray.dequantize(qa_tensor)

        # Compute mean absolute error in float32 to avoid bfloat16 precision loss.
        x_f32 = x.astype(jnp.float32)
        error_block = jnp.abs(x_f32 - x_block.astype(jnp.float32)).mean()
        error_tensor = jnp.abs(x_f32 - x_tensor.astype(jnp.float32)).mean()
        self.assertLess(float(error_block), float(error_tensor))


class ChannelwiseTileDotGeneralTest(unittest.TestCase):
    """Test dot_general with channelwise tiling through the full QDQ pipeline."""

    def test_blockwise_fp8_dot_general(self):
        """Block-wise FP8 dot_general should produce output close to bf16."""
        key = jax.random.PRNGKey(0)
        lhs = jax.random.normal(key, (8, 512), dtype=jnp.bfloat16)
        rhs = jax.random.normal(key, (512, 256), dtype=jnp.bfloat16)
        dnums = (((1,), (0,)), ((), ()))

        config = dot_general_qt.DotGeneralQtConfig(
            lhs_qtype=jnp.float8_e4m3fn,
            rhs_qtype=jnp.float8_e4m3fn,
            tile_size=128,
            lhs_channelwise_tile_size=None,
            rhs_channelwise_tile_size=128,
        )

        y_bf16 = jax.lax.dot_general(lhs, rhs, dnums)
        y_fp8 = dot_general_qt.dot_general_qt(lhs, rhs, dnums, config)

        rel_error = jnp.abs(y_bf16 - y_fp8).mean() / (jnp.abs(y_bf16).mean() + 1e-8)
        self.assertLess(float(rel_error), 0.05)

    def test_blockwise_fp8_gradient(self):
        """Gradients through block-wise FP8 dot_general should be close to bf16."""
        key = jax.random.PRNGKey(0)
        lhs = jax.random.normal(key, (8, 512), dtype=jnp.bfloat16)
        rhs = jax.random.normal(key, (512, 256), dtype=jnp.bfloat16)
        dnums = (((1,), (0,)), ((), ()))

        config = dot_general_qt.DotGeneralQtConfig(
            lhs_qtype=jnp.float8_e4m3fn,
            rhs_qtype=jnp.float8_e4m3fn,
            tile_size=128,
            lhs_channelwise_tile_size=None,
            rhs_channelwise_tile_size=128,
            dlhs_grad_qtype=jnp.float8_e4m3fn,
            drhs_grad_qtype=jnp.float8_e4m3fn,
        )

        def f_bf16(l, r):
            return jax.lax.dot_general(l, r, dnums).sum()

        def f_fp8(l, r):
            return dot_general_qt.dot_general_qt(l, r, dnums, config).sum()

        grad_bf16 = jax.grad(f_bf16, argnums=(0, 1))(lhs, rhs)
        grad_fp8 = jax.grad(f_fp8, argnums=(0, 1))(lhs, rhs)

        for g_ref, g_fp8 in zip(grad_bf16, grad_fp8):
            rel_error = jnp.abs(g_ref - g_fp8).mean() / (jnp.abs(g_ref).mean() + 1e-8)
            self.assertLess(float(rel_error), 0.1)


if __name__ == "__main__":
    unittest.main()
