# pylint: disable=no-name-in-module
"""Regression coverage for FIA parser/replay helpers.

The JSONL enrichment utility was removed, but this file remains as the
regression coverage home for FIA metadata parsing and replay inference helpers.
"""

import csv

from tools.perf_data_collection import fia_common
from tools.perf_data_collection.op_replay.FusedInferAttentionScore_run import (
    build_scalar_length_list,
    cumulative_lengths,
    distribute_total,
    infer_block_size,
    infer_case_args,
    infer_kv_block_shape,
    infer_query_lens,
    infer_seq_lens_kv,
    infer_sparse_mode,
    resolve_input_shapes,
    validate_case_for_replay,
)
from tools.perf_data_collection.parsers.parse_kernel_details import (
    EXTRA_NUMERIC_COLUMNS,
    FiaRuntimeMetadata,
    KernelDetailsParser,
    extract_fia_profile_metadata,
    infer_avg_seq_len,
    profiling_column_name,
)


def _base_kernel_row(input_shapes: str, output_shapes: str) -> dict[str, str]:
    row = {
        "Type": "FusedInferAttentionScore",
        "OP State": "dynamic",
        "Accelerator Core": "MIX_AIC",
        "Input Shapes": input_shapes,
        "Input Data Types": "DT_BF16;DT_BF16;DT_BF16",
        "Input Formats": "ND;ND;ND",
        "Output Shapes": output_shapes,
        "Output Data Types": "DT_BF16;FLOAT",
        "Output Formats": "ND;ND",
        "Duration(us)": "1.0",
    }
    for column in EXTRA_NUMERIC_COLUMNS:
        row[column] = "0"
    return row


def _build_input_shapes(
    *,
    query: str = "5,16,1,512",
    key: str = "1171,1,128,512",
    value: str = "1171,1,128,512",
    seq: str = "5",
    seq_kv: str = "5",
    block: str = "5,512",
    query_rope: str = "5,16,1,64",
    key_rope: str = "1171,1,128,64",
) -> str:
    slots = [""] * 31
    slots[0] = query
    slots[1] = key
    slots[2] = value
    slots[5] = seq
    slots[6] = seq_kv
    slots[14] = block
    slots[24] = query_rope
    slots[25] = key_rope
    return ";".join(slots)


class TestFiaCommonHelpers:
    def test_parse_runtime_metadata_fields(self):
        assert fia_common.split_metadata_field('"1,2; ;3,4"') == ["1,2", "", "3,4"]
        assert fia_common.parse_shape_or_none("1, 2,3") == (1, 2, 3)
        assert fia_common.parse_shape_or_none(" ") is None
        assert fia_common.parse_runtime_int(" 16 ") == 16
        assert fia_common.parse_runtime_int("") is None
        assert fia_common.parse_runtime_int_list("1; 2,3") == [1, 2, 3]
        assert fia_common.parse_runtime_int_list("") is None
        assert fia_common.shape_numel((2, 3, 4)) == 24
        assert fia_common.shape_numel(None) == 0
        assert fia_common.shape_to_text((1, 2, 3)) == "1,2,3"
        assert fia_common.shape_to_text(None) == ""


class TestFiaReplayInference:
    def test_fia_replay_prefers_operator_raw_shapes_for_mla(self):
        row = {
            "Input Shapes": (
                "4099,16,128;4099,16,128;4099,16,128;;2048,2048;1;1"
                ";;;;;;;;;;;;1,512;;;;;;;;;;4099,16,64;4099,16,64;;;;;"
            ),
            "Runtime operator_input_shapes_raw": (
                "16,1,1,512;4099,1,128,512;4099,1,128,512;;;1;1;;;;;;;;1,512;;;;;;;;;;16,1,1,64;4099,1,128,64;;;;;"
            ),
            "Runtime input_layout": "BNSD_NBSD",
            "Runtime num_heads": "16",
            "Runtime num_key_value_heads": "1",
        }

        input_shapes = resolve_input_shapes(row)
        inferred = infer_case_args(input_shapes, row)

        assert input_shapes[0] == (16, 1, 1, 512)
        assert input_shapes[1] == (4099, 1, 128, 512)
        assert input_shapes[24] == (16, 1, 1, 64)
        assert input_shapes[25] == (4099, 1, 128, 64)
        assert inferred["input_layout"] == "BNSD_NBSD"
        assert inferred["num_heads"] == 16
        assert inferred["num_key_value_heads"] == 1

    def test_fia_replay_infers_lengths_and_modes(self):
        assert distribute_total(10, 3, min_value=1) == [4, 3, 3]
        assert cumulative_lengths([4, 3, 3]) == [4, 7, 10]
        assert build_scalar_length_list((2, 3)) == 6
        assert build_scalar_length_list(None) is None
        assert infer_sparse_mode(None, {}) == 0
        assert infer_sparse_mode((2048, 2048), {}) == 3
        assert infer_sparse_mode(None, {"Runtime sparse_mode": "4"}) == 4
        assert infer_block_size((100, 16, 128), (2, 8), {}) == 16
        assert infer_block_size((100, 1, 128, 512), (2, 8), {}) == 128
        assert infer_block_size((100, 1, 128, 512), (2, 8), {"Runtime block_size": "64"}) == 64
        assert infer_query_lens((9, 16, 128), 3) == [3, 6, 9]
        assert infer_query_lens((3, 16, 4, 128), 3) == [4, 4, 4]
        assert infer_kv_block_shape((100, 16, 128), (2, 8)) == (100, 16)
        assert infer_kv_block_shape((100, 1, 128, 512), (2, 8)) == (100, 128)

    def test_fia_replay_uses_runtime_seq_lens_when_available(self):
        assert infer_seq_lens_kv(
            (100, 16, 128),
            batch_size=2,
            block_table_shape=(2, 8),
            runtime_row={"Runtime actual_seq_lengths_kv_values": "40,998"},
        ) == [40, 998]
        assert infer_seq_lens_kv(
            (100, 16, 128),
            batch_size=2,
            block_table_shape=(2, 8),
            runtime_row={"Runtime block_table_valid_blocks": "2,3"},
        ) == [32, 48]

    def test_fia_replay_rejects_mla_row_without_raw_shapes(self):
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape
                self.ndim = len(shape)

        row = {
            "Runtime num_key_value_heads": "1",
            "Runtime input_layout": "TND",
            "Runtime operator_input_shapes_raw": "",
        }
        case = {
            "key": FakeTensor((4099, 16, 128)),
            "input_layout": "TND",
            "num_key_value_heads": 1,
        }

        error = validate_case_for_replay(case, row)
        assert error is not None
        assert "Runtime operator_input_shapes_raw is missing" in error

    def test_fia_replay_accepts_standard_tnd_row_without_raw_shapes(self):
        class FakeTensor:
            def __init__(self, shape):
                self.shape = shape
                self.ndim = len(shape)

        row = {
            "Runtime num_key_value_heads": "16",
            "Runtime input_layout": "TND",
            "Runtime operator_input_shapes_raw": " ",
        }
        case = {
            "key": FakeTensor((4099, 16, 128)),
            "input_layout": "TND",
            "num_key_value_heads": 16,
        }

        assert validate_case_for_replay(case, row) is None


class TestKernelDetailsParserFiaHelpers:
    def test_parser_static_helpers(self):
        assert profiling_column_name("Duration(us)") == "Profiling Duration(us)"
        assert infer_avg_seq_len("40, 998") == "519.000000"
        assert infer_avg_seq_len("bad") == ""
        assert KernelDetailsParser._parse_duration("1.25") == 1.25
        assert KernelDetailsParser._parse_duration("bad") == 0.0
        assert KernelDetailsParser._sanitize_filename('A/B:*?"') == "A_B____"
        assert KernelDetailsParser._safe_cell({"a": " x "}, "a") == "x"
        assert KernelDetailsParser._is_na_shape('"N/A"')
        assert KernelDetailsParser._normalize_kernel_type("muls_add_kernel_1") == "muls_add_kernel"
        assert KernelDetailsParser._shape_key({"Input Shapes": "1,2", "Output Shapes": "3"}) == ("1,2", "3")
        assert KernelDetailsParser._normalize_text("Fused Infer_Attention-Score") == "fusedinferattentionscore"
        assert KernelDetailsParser._is_fia_operator_row({"Name": "aclnnFusedInferAttentionScoreV2"})

    def test_extract_fia_profile_metadata_from_slots(self):
        metadata = extract_fia_profile_metadata(
            input_shapes_text=_build_input_shapes(seq="5", seq_kv="5", block="5,512"),
            source_profile="PROF_001",
        )

        assert metadata.source_profile == "PROF_001"
        assert metadata.actual_seq_lengths_shape == "5"
        assert metadata.actual_seq_lengths_kv_shape == "5"
        assert metadata.block_table_shape == "5,512"
        assert metadata.metadata_completeness == "profile_shapes_only"

    def test_compute_fia_metadata_completeness(self):
        metadata = FiaRuntimeMetadata(
            source_profile="p",
            actual_seq_lengths_shape="",
            actual_seq_lengths_values="",
            actual_seq_lengths_kv_shape="",
            actual_seq_lengths_kv_values="",
            avg_seq_len="",
            block_table_shape="",
            block_table_valid_blocks="",
            num_heads="",
            num_key_value_heads="",
            sparse_mode="",
            input_layout="",
            block_size="",
            attn_state="",
            kv_cache_mode="",
            metadata_completeness="profile_shapes_only",
        )
        assert KernelDetailsParser._compute_fia_metadata_completeness(metadata) == "profile_shapes_only"
        metadata.actual_seq_lengths_kv_values = "40,998"
        assert KernelDetailsParser._compute_fia_metadata_completeness(metadata) == "runtime_values"

    def test_parse_kernel_details_adds_profile_shapes_metadata(self, tmp_path):
        profile_dir = tmp_path / "profile_a"
        profile_dir.mkdir()

        kernel_row = _base_kernel_row(
            _build_input_shapes(
                query="8192,16,128",
                key="8192,16,128",
                value="8192,16,128",
                seq="2",
                seq_kv="2",
                block="",
                query_rope="8192,16,64",
                key_rope="8192,16,64",
            ),
            "8192,16,128;8192,16,1",
        )
        kernel_csv = profile_dir / "kernel_details.csv"
        with kernel_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kernel_row.keys()))
            writer.writeheader()
            writer.writerow(kernel_row)

        operator_row = {
            "Type": "aclnnFusedInferAttentionScoreV2",
            "Name": "npu_fused_infer_attention_score_v2",
            "Input Shapes": _build_input_shapes(
                query="5,16,1,512",
                key="1171,1,128,512",
                value="1171,1,128,512",
                seq="5",
                seq_kv="5",
                block="5,512",
                query_rope="5,16,1,64",
                key_rope="1171,1,128,64",
            ),
            "Output Shapes": "5,16,1,512;5,16,1,1",
        }
        operator_csv = profile_dir / "operator_details.csv"
        with operator_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(operator_row.keys()))
            writer.writeheader()
            writer.writerow(operator_row)

        parser = KernelDetailsParser(
            device="TEST_DEVICE",
            kernel_details_path=str(tmp_path),
            database_path=tmp_path / "db_out",
        )
        output_files = parser.parse_and_export()

        output_csv = next(path for path in output_files if path.name == "FusedInferAttentionScore.csv")
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))

        assert row["Runtime source_profile"] == "profile_a"
        assert row["Runtime actual_seq_lengths_shape"] == "5"
        assert row["Runtime actual_seq_lengths_kv_shape"] == "5"
        assert row["Runtime block_table_shape"] == "5,512"
        assert row["Runtime metadata_completeness"] == "profile_shapes_only"
