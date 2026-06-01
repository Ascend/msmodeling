import unittest

import pytest
import torch
from parameterized import parameterized
from tensor_cast.core.quantization.datatypes import QuantizeLinearAction
from tensor_cast.core.user_config import UserInputConfig
from tensor_cast.device import TEST_DEVICE
from tensor_cast.layers.mla import MultiheadLatentAttentionTensorCast
from tensor_cast.layers.quant_linear import QuantLinearBase, TensorCastQuantLinear
from tensor_cast.layers.sampler import SamplingMetadata
from tensor_cast.model_config import LinearQuantConfig, MlaConfig, ModelConfig, MtpConfig, ParallelConfig, QuantConfig
from tensor_cast.patch_torch import patch_torch
from tensor_cast.performance_model.analytic import AnalyticPerformanceModel
from tensor_cast.quantize_utils import LinearQuantType, QuantGranularity, QuantScheme
from tensor_cast.runtime import Runtime
from tensor_cast.transformers.custom_model_registry import get_moe_config, get_mtp_block_module_name
from tensor_cast.transformers.model import TransformerModel
from tensor_cast.transformers.utils import AutoModelConfigLoader
from tensor_cast.utils import DTYPE_FP8

from .test_common import (
    create_mla_metadata_and_kv_cache,
    get_cached_build_model,
    get_linear_quant_config,
    get_quant_config,
    has_submodule_with_cls_name,
)

# Core int4 roundtrip and FP8 configuration assertions were moved to matching tests in test_dtype.py.

# Define common parameters for tests
IN_FEATURES = 32
OUT_FEATURES = 64
BATCH_SIZE = 4
MODEL_DTYPE = torch.bfloat16
DEVICE = "cpu"


class QuantLinearTestMixin:
    @classmethod
    def setUpClass(cls):
        cls._model_cache = {}

    @classmethod
    def _get_model(cls, user_config: UserInputConfig):
        return get_cached_build_model(cls._model_cache, user_config)

    def setUp(self):
        """Set up common resources for tests."""
        torch.compiler.reset()


class TestQuantLinear(QuantLinearTestMixin, unittest.TestCase):
    def setUp(self):
        """Set up common resources for tests."""
        super().setUp()
        torch.manual_seed(0)
        self.linear_layer_with_bias = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=True).to(
            DEVICE, dtype=MODEL_DTYPE
        )

        torch.manual_seed(0)
        self.linear_layer_no_bias = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False).to(DEVICE, dtype=MODEL_DTYPE)

        torch.manual_seed(1)
        self.input_tensor = torch.randn(BATCH_SIZE, IN_FEATURES).to(DEVICE, dtype=MODEL_DTYPE)

    # deprecated: migrated to test_dtype.py::test_pack_unpack_int4_roundtrip
    def test_pack_unpack_int4_roundtrip(self):
        """Tests if packing and then unpacking an int4 tensor restores the original."""
        original_tensor = torch.randint(-8, 8, (OUT_FEATURES, IN_FEATURES), dtype=torch.int8, device=DEVICE)
        dummy_layer = QuantLinearBase(
            self.linear_layer_no_bias,
            get_linear_quant_config(LinearQuantType.W4A8, torch.randn(1)),
        )

        packed_dim1 = dummy_layer.pack_int4(original_tensor)
        unpacked_dim1 = dummy_layer.unpack_int4(packed_dim1)
        self.assertTrue(torch.equal(original_tensor, unpacked_dim1))

    def test_dequantize_weight(self):
        """Tests that dequantized weight is close to the original float weight."""
        original_weight = self.linear_layer_with_bias.weight.data

        # W8 Symmetric per-tensor quantization
        config = get_linear_quant_config(LinearQuantType.W8A16, original_weight)
        quant_layer = QuantLinearBase(self.linear_layer_with_bias, config)

        dequantized_weight = quant_layer.dequantize_weight()

        # NOTE: we only check shapes
        self.assertEqual(dequantized_weight.shape, original_weight.shape)

    def test_forward_pass_equivalence(self):
        """Tests the forward pass against a standard nn.Linear layer for various configs."""
        test_configs = [
            {
                "quant_type": LinearQuantType.W8A16.value,
                "use_bias": True,
                "w_scheme": "symmetric",
                "a_scheme": None,
            },
            {
                "quant_type": LinearQuantType.W8A16.value,
                "use_bias": False,
                "w_scheme": "asymmetric",
                "a_scheme": None,
            },
            {
                "quant_type": LinearQuantType.W8A8.value,
                "use_bias": True,
                "w_scheme": "symmetric",
                "a_scheme": "symmetric",
            },
            {
                "quant_type": LinearQuantType.W8A8.value,
                "use_bias": True,
                "w_scheme": "asymmetric",
                "a_scheme": "asymmetric",
            },
            {
                "quant_type": LinearQuantType.W4A8.value,
                "use_bias": True,
                "w_scheme": "symmetric",
                "a_scheme": "symmetric",
            },
            {
                "quant_type": LinearQuantType.W4A8.value,
                "use_bias": False,
                "w_scheme": "symmetric",
                "a_scheme": "symmetric",
            },
        ]

        for params in test_configs:
            with self.subTest(**params):
                torch.manual_seed(42)
                linear_layer = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=params["use_bias"]).to(
                    DEVICE, dtype=MODEL_DTYPE
                )
                weight = linear_layer.weight.data

                config_kwargs = {}
                if params["w_scheme"] == "symmetric":
                    w_scale = torch.max(torch.abs(weight)) / 127.0
                    config_kwargs["weight_scale"] = w_scale
                else:  # asymmetric
                    w_min, w_max = torch.min(weight), torch.max(weight)
                    w_scale = (w_max - w_min) / 255.0
                    w_offset = -128.0 - w_min / w_scale
                    config_kwargs["weight_scale"] = w_scale
                    config_kwargs["weight_offset"] = w_offset

                if params["a_scheme"]:
                    config_kwargs["dynamic_quant_scheme"] = (
                        QuantScheme.SYMMETRIC if params["a_scheme"] == "symmetric" else QuantScheme.ASYMMETRIC
                    )

                quant_type_enum = LinearQuantType(params["quant_type"])

                config = get_linear_quant_config(quant_type_enum, weight, **config_kwargs)
                quant_linear_layer = QuantLinearBase(linear_layer, config)

                expected_output = linear_layer(self.input_tensor)
                actual_output = quant_linear_layer(self.input_tensor)

                # NOTE: we only check shapes
                self.assertEqual(actual_output.shape, expected_output.shape)
                self.assertEqual(actual_output.dtype, MODEL_DTYPE)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_wildcard(self, model_id):
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        model_config_with_quant = ModelConfig(
            ParallelConfig(),
            get_quant_config(),
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        qmodel = TransformerModel(model_id, model_config_with_quant)
        num_linear_modules = sum(1 for _, module in qmodel.named_modules() if isinstance(module, torch.nn.Linear))
        # lm_head will never be quantized
        self.assertEqual(num_linear_modules, 1)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with (
            torch.no_grad(),
            patch_torch(),
        ):
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.vocab_size))

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_base(self, model_id):
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)
        num_linear_modules = sum(1 for _, module in model.named_modules() if isinstance(module, torch.nn.Linear))

        model_config_with_quant = ModelConfig(
            ParallelConfig(),
            get_quant_config(model.unwrap()),
            quant_linear_cls=QuantLinearBase,
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        qmodel = TransformerModel(model_id, model_config_with_quant)
        num_qlinear_modules = sum(1 for _, module in qmodel.named_modules() if isinstance(module, QuantLinearBase))
        # lm_head will never be quantized
        self.assertEqual(num_qlinear_modules + 1, num_linear_modules)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        with (
            torch.no_grad(),
            patch_torch(),
        ):
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.vocab_size))

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B", True, False],
            ["Qwen/Qwen3-235B-A22B", True, True],
            ["zai-org/GLM-4.5", True, False],
            ["Qwen/Qwen3-32B", False, True],
        ]
    )
    def test_model_quant_tensorcast_dynamic_w4a8(self, model_id, symmetric, per_sample):
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            moe_config=moe_config,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)

        model_config_with_quant = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                model.unwrap(),
                quant_type=LinearQuantType.W4A8,
                dynamic_quant_scheme=QuantScheme.SYMMETRIC if symmetric else QuantScheme.ASYMMETRIC,
                dynamic_quant_granularity=QuantGranularity.PER_SAMPLE if per_sample else QuantGranularity.PER_TENSOR,
            ),
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        qmodel = TransformerModel(model_id, model_config_with_quant)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.vocab_size))
        result = runtime.table_averages()
        if symmetric:
            self.assertIn("tensor_cast.dynamic_quantize_symmetric.default", result)
        else:
            self.assertIn("tensor_cast.dynamic_quantize_asymmetric.default", result)
        self.assertIn("tensor_cast.static_quant_linear_int4.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_tensorcast_static_w8a8(self, model_id):
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        model_config = ModelConfig(
            ParallelConfig(),
            QuantConfig(),
            moe_config=moe_config,
            hf_config=hf_config,
        )
        model = TransformerModel(model_id, model_config)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        model_config_with_quant = ModelConfig(
            ParallelConfig(),
            get_quant_config(
                model.unwrap(),
                quant_type=LinearQuantType.W8A8,
                activation_scale=torch.empty([num_tokens], dtype=torch.float, device="meta"),
            ),
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        qmodel = TransformerModel(model_id, model_config_with_quant)
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.vocab_size))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.quantize.default", result)
        self.assertIn("tensor_cast.static_quant_linear.default", result)

    def _run_test_deepseek_mtp_quant_tensorcast_static_w8a8(self, model_id, do_compile):
        num_mtp_layers = 1
        user_config = UserInputConfig(
            model_id=model_id,
            num_mtp_tokens=num_mtp_layers,
            quantize_linear_action=QuantizeLinearAction.W8A8_STATIC,
            do_compile=do_compile,
        )

        model = self._get_model(user_config)

        mtp_block_module_name = get_mtp_block_module_name(model.model_config.hf_config.model_type)
        self.assertIsNotNone(mtp_block_module_name)

        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model.model_config)
        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))

        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with Runtime(perf_model, machine_config) as runtime, torch.no_grad():
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(query_start_loc=attn_meta.query_start_loc),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.quantize.default", result)
        self.assertIn("tensor_cast.static_quant_linear.default", result)

    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1", False],
        ]
    )
    def test_deepseek_mtp_quant_tensorcast_static_w8a8(self, model_id, do_compile):
        self._run_test_deepseek_mtp_quant_tensorcast_static_w8a8(model_id, do_compile)

    def test_quantize_lmhead(self):
        model_id = "Qwen/Qwen3-32B"
        linear_quant_config = get_linear_quant_config(
            LinearQuantType.W8A8,
            torch.randn(1),
        )
        quant_config = QuantConfig()
        quant_config.linear_configs["lm_head"] = linear_quant_config
        quant_config.modules_to_not_convert = []
        model_config = ModelConfig(
            ParallelConfig(),
            quant_config,
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
        )
        model = TransformerModel(model_id, model_config)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            _ = model.forward(inputs, position_ids)
        result = runtime.table_averages()
        self.assertIn("tensor_cast.dynamic_quantize_symmetric.default", result)
        self.assertIn("tensor_cast.static_quant_linear.default", result)

    def test_quantize_lmhead_mtp(self):
        model_id = "deepseek-ai/DeepSeek-V3.1"
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        linear_quant_config = get_linear_quant_config(
            LinearQuantType.W8A8,
            torch.randn(1),
        )
        quant_config = QuantConfig()
        quant_config.linear_configs["*.lm_head"] = linear_quant_config
        quant_config.modules_to_not_convert = []
        model_config = ModelConfig(
            ParallelConfig(),
            quant_config,
            quant_linear_cls=TensorCastQuantLinear,
            enable_repetition=True,
            moe_config=moe_config,
            hf_config=hf_config,
            trust_remote_code=False,
        )
        mla_config = MlaConfig(
            module_name="DeepseekV3Attention",
            mla_cls=MultiheadLatentAttentionTensorCast,
        )
        model_config.mla_config = mla_config
        num_mtp_layers = 1
        mtp_block_module_name = get_mtp_block_module_name(hf_config.model_type)
        self.assertIsNotNone(mtp_block_module_name)
        mtp_config = MtpConfig(
            num_mtp_layers=num_mtp_layers,
            mtp_block_module_name=mtp_block_module_name,
        )
        model_config.mtp_config = mtp_config
        model = TransformerModel(model_id, model_config)
        # make sure all original attention modules have been replaced
        self.assertTrue(has_submodule_with_cls_name(model, "MultiheadLatentAttentionTensorCast"))
        attn_meta, kv_cache_by_layers, num_tokens = create_mla_metadata_and_kv_cache(model, model_config)
        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)
        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            outputs = model.forward(
                inputs,
                position_ids,
                attention_meta=attn_meta,
                kv_cache_by_layers=kv_cache_by_layers,
                sampling_metadata=SamplingMetadata(query_start_loc=attn_meta.query_start_loc),
            )
            self.assertEqual(outputs.shape, (2, num_mtp_layers + 1))
        result = runtime.table_averages()
        self.assertIn("tensor_cast.dynamic_quantize_symmetric.default", result)
        self.assertIn("tensor_cast.static_quant_linear.default", result)

    # deprecated (asymmetric branch): migrated to test_dtype.py::test_fp8_rejects_asymmetric_scheme
    def test_fp8_validation(self):
        """Test that FP8 quantization validates configuration correctly."""
        # Test valid FP8 configuration
        valid_config = get_linear_quant_config(
            quant_type=LinearQuantType.FP8,
            weight_scale=torch.tensor(1.0),
            dynamic_quant_scheme=QuantScheme.SYMMETRIC,
        )
        # This should not raise an error
        fp8_layer = QuantLinearBase(self.linear_layer_no_bias, valid_config)
        self.assertEqual(fp8_layer.quant_config.quant_type, LinearQuantType.FP8)

        # Test invalid configurations

        # 1. Asymmetric scheme not allowed
        with self.assertRaises(ValueError) as cm:
            invalid_config = LinearQuantConfig(
                quant_type=LinearQuantType.FP8,
                weight_scale=torch.tensor(1.0),
                dynamic_quant_scheme=QuantScheme.ASYMMETRIC,
            )
            QuantLinearBase(self.linear_layer_no_bias, invalid_config)
        self.assertIn("symmetric scheme", str(cm.exception))

        # 2. Static activation quantization not allowed
        with self.assertRaises(ValueError) as cm:
            invalid_config = LinearQuantConfig(
                quant_type=LinearQuantType.FP8,
                weight_scale=torch.tensor(1.0),
                activation_scale=torch.tensor(1.0),
            )
            QuantLinearBase(self.linear_layer_no_bias, invalid_config)
        self.assertIn("static activation", str(cm.exception))

    def test_fp8_forward_pass_base(self):
        """Test FP8 forward pass in QuantLinearBase."""
        config = get_linear_quant_config(
            quant_type=LinearQuantType.FP8,
            weight_scale=torch.tensor(1.0),
            dynamic_quant_scheme=QuantScheme.SYMMETRIC,
        )
        fp8_layer = QuantLinearBase(self.linear_layer_with_bias, config)

        # Test forward pass
        output = fp8_layer(self.input_tensor)

        # Check output shape and dtype
        expected_shape = (BATCH_SIZE, OUT_FEATURES)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, MODEL_DTYPE)

    @parameterized.expand(
        [
            [True, True],  # per_sample=True, with_bias=True
            [True, False],  # per_sample=True, with_bias=False
            [False, True],  # per_sample=False, with_bias=True
            [False, False],  # per_sample=False, with_bias=False
        ]
    )
    def test_fp8_tensorcast_dynamic(self, per_sample, with_bias):
        """Test FP8 dynamic quantization with TensorCastQuantLinear."""
        # Create linear layer with or without bias
        linear_layer = torch.nn.Linear(IN_FEATURES, OUT_FEATURES, bias=with_bias).to(DEVICE, dtype=MODEL_DTYPE)

        config = get_linear_quant_config(
            quant_type=LinearQuantType.FP8,
            weight_scale=torch.tensor(1.0),
            dynamic_quant_scheme=QuantScheme.SYMMETRIC,
            dynamic_quant_granularity=QuantGranularity.PER_SAMPLE if per_sample else QuantGranularity.PER_TENSOR,
        )

        fp8_layer = TensorCastQuantLinear(linear_layer, config)

        # Test forward pass
        output = fp8_layer(self.input_tensor.to("meta"))

        # Check output shape and dtype
        expected_shape = (BATCH_SIZE, OUT_FEATURES)
        self.assertEqual(output.shape, expected_shape)

        # Verify the layer uses FP8 operations
        self.assertEqual(fp8_layer.quant_config.quant_type, LinearQuantType.FP8)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_tensorcast_fp8(self, model_id):
        """Test FP8 quantization on full model."""
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        # Create FP8 quantization config
        fp8_quant_config = get_quant_config(
            quant_type=LinearQuantType.FP8,
        )
        model_config_with_fp8 = ModelConfig(
            ParallelConfig(),
            fp8_quant_config,
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        qmodel = TransformerModel(model_id, model_config_with_fp8)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)

        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.vocab_size))

        result = runtime.table_averages()
        # Check that FP8 operations are being used
        self.assertIn("tensor_cast.dynamic_quantize_symmetric.default", result)
        self.assertIn("tensor_cast.fp8_linear.default", result)

    @parameterized.expand(
        [
            ["Qwen/Qwen3-32B"],
            ["Qwen/Qwen3-235B-A22B"],
            ["zai-org/GLM-4.5"],
        ]
    )
    def test_model_quant_tensorcast_mxfp4(self, model_id):
        """Test MXFP4 quantization on full model."""
        # Create MXFP4 quantization config
        # MXFP4 requires per-channel-group quantization.
        # Let's assume a group size that results in K_group=64 for the scale.
        mxfp4_quant_config = get_quant_config(
            quant_type=LinearQuantType.MXFP4,
            weight_group_size=32,
            weight_quant_granularity=QuantGranularity.PER_GROUP,
            weight_quant_scheme=QuantScheme.SYMMETRIC,
        )
        auto_loader = AutoModelConfigLoader()
        hf_config = auto_loader.load_config(model_id)
        moe_config = get_moe_config(hf_config.model_type)
        model_config_with_mxfp4 = ModelConfig(
            ParallelConfig(),
            mxfp4_quant_config,
            quant_linear_cls=TensorCastQuantLinear,
            num_hidden_layers_override=2,
            moe_config=moe_config,
            hf_config=hf_config,
        )
        qmodel = TransformerModel(model_id, model_config_with_mxfp4)

        num_tokens = 100
        inputs = torch.empty([1, num_tokens], dtype=torch.long, device="meta")
        position_ids = torch.empty([1, num_tokens], dtype=torch.long, device="meta")

        machine_config = TEST_DEVICE
        perf_model = AnalyticPerformanceModel(machine_config)

        with (
            Runtime(perf_model, machine_config) as runtime,
            torch.no_grad(),
        ):
            outputs = qmodel.forward(inputs, position_ids)
            self.assertEqual(outputs.shape, (1, num_tokens, qmodel.vocab_size))

        result = runtime.table_averages()
        # Check that MXFP4 operations are being used
        self.assertIn("tensor_cast.dynamic_quantize_mxfp4.default", result)
        self.assertIn("tensor_cast.mxfp4_linear.default", result)

    def test_fp8_weight_quantization(self):
        """Test that FP8 weights are properly quantized during initialization."""
        config = LinearQuantConfig(
            quant_type=LinearQuantType.FP8,
            weight_scale=torch.tensor(1.0),
            dynamic_quant_scheme=QuantScheme.SYMMETRIC,
        )

        fp8_layer = QuantLinearBase(self.linear_layer_with_bias, config)

        # Check that quantized weights have FP8 dtype
        self.assertEqual(fp8_layer.qweight.dtype, DTYPE_FP8)

        # Check that bias is preserved
        if self.linear_layer_with_bias.bias is not None:
            self.assertIsNotNone(fp8_layer.bias)
            self.assertEqual(fp8_layer.bias.shape, self.linear_layer_with_bias.bias.shape)


@pytest.mark.nightly
class TestQuantLinearNightly(QuantLinearTestMixin, unittest.TestCase):
    @parameterized.expand(
        [
            ["deepseek-ai/DeepSeek-V3.1"],
            ["moonshotai/Kimi-K2-Base"],
        ]
    )
    def test_deepseek_mtp_quant_tensorcast_static_w8a8(self, model_id):
        TestQuantLinear._run_test_deepseek_mtp_quant_tensorcast_static_w8a8(self, model_id, True)

    @parameterized.expand(
        [
            ["moonshotai/Kimi-K2-Base"],
        ]
    )
    def test_deepseek_mtp_quant_tensorcast_static_w8a8_kimi(self, model_id):
        TestQuantLinear._run_test_deepseek_mtp_quant_tensorcast_static_w8a8(self, model_id, False)
