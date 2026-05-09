import csv

from tools.perf_data_collection.fill_fia_runtime_metadata import (
    backfill,
    build_csv_key,
    build_jsonl_key,
)
from tools.perf_data_collection.op_replay.FusedInferAttentionScore_run import (
    infer_case_args,
    resolve_input_shapes,
    validate_case_for_replay,
)
from tools.perf_data_collection.parsers.parse_kernel_details import (
    EXTRA_NUMERIC_COLUMNS,
    KernelDetailsParser,
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
    *, query: str, key: str, value: str, seq_kv_len: str, block: str
) -> str:
    slots = [""] * 15
    slots[0] = query
    slots[1] = key
    slots[2] = value
    slots[6] = seq_kv_len
    slots[14] = block
    return ";".join(slots)


class TestFiaOperatorDetailsEnrichment:
    def test_fia_replay_prefers_operator_raw_shapes_for_mla(self):
        row = {
            "Input Shapes": (
                "4099,16,128;4099,16,128;4099,16,128;;2048,2048;1;1"
                ";;;;;;;;;;;;1,512;;;;;;;;;;4099,16,64;4099,16,64;;;;;"
            ),
            "Runtime operator_input_shapes_raw": (
                "16,1,1,512;4099,1,128,512;4099,1,128,512;;;"
                "1;1;;;;;;;;1,512;;;;;;;;;;16,1,1,64;4099,1,128,64;;;;;"
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

        error = validate_case_for_replay(case, row)
        assert error is None

    def test_parse_kernel_details_prefers_operator_raw_shapes_for_mla(self, tmp_path):
        profile_dir = tmp_path / "profile_a"
        profile_dir.mkdir()

        kernel_csv = profile_dir / "kernel_details.csv"
        operator_csv = profile_dir / "operator_details.csv"

        kernel_row = _base_kernel_row(
            (
                "8192,16,128;8192,16,128;8192,16,128;;2048,2048;2;2"
                ";;;;;;;;;;;;;;;;;;8192,16,64;8192,16,64;;;;;"
            ),
            "8192,16,128;8192,16,1",
        )

        with kernel_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(kernel_row.keys()))
            writer.writeheader()
            writer.writerow(kernel_row)

        operator_row = {
            "Type": "aclnnFusedInferAttentionScoreV2",
            "Name": "npu_fused_infer_attention_score_v2",
            "Input Shapes": (
                "5,16,1,512;1171,1,128,512;1171,1,128,512;"
                ";;;;;;;;;;;;5,512;;;;;;;;;;5,16,1,64;1171,1,128,64"
            ),
            "Output Shapes": "5,16,1,512;5,16,1,1",
        }
        with operator_csv.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(operator_row.keys()))
            writer.writeheader()
            writer.writerow(operator_row)

        parser = KernelDetailsParser(
            device="TEST_DEVICE",
            kernel_details_path=str(tmp_path),
            database_path=tmp_path / "db_out",
        )
        parser.output_dir = tmp_path / "out"
        output_files = parser.parse_and_export()

        output_name = "FusedInferAttentionScore.csv"
        output_csv = next(path for path in output_files if path.name == output_name)
        with output_csv.open("r", encoding="utf-8", newline="") as handle:
            row = next(csv.DictReader(handle))

        assert "Runtime operator_input_shapes_raw" not in row
        assert row["Runtime block_table_shape"] == ""
        assert row["Runtime actual_seq_lengths_shape"] == ""
        assert row["Runtime actual_seq_lengths_kv_shape"] == ""
        assert row["Runtime input_layout"] == ""
        assert row["Runtime num_key_value_heads"] == ""
        assert row["Runtime attn_state"] == ""
        assert row["Runtime metadata_completeness"] == "profile_shapes_only"


class TestFiaBackfillSignature:
    def test_build_keys_match_for_same_runtime_case(self):
        csv_row = {
            "Input Shapes": _build_input_shapes(
                query="5,16,1,512",
                key="1171,1,128,512",
                value="1171,1,128,512",
                seq_kv_len="1",
                block="5,512",
            ),
        }
        json_record = {
            "query_shape": [5, 16, 1, 512],
            "key_shape": [1171, 1, 128, 512],
            "value_shape": [1171, 1, 128, 512],
            "actual_seq_lengths_kv": [1024],
            "atten_mask_shape": None,
            "block_table_shape": [5, 512],
        }

        assert build_csv_key(csv_row) == build_jsonl_key(json_record)

    def test_backfill_expands_all_runtime_variants_for_same_key(self):
        rows = [
            {
                "Input Shapes": _build_input_shapes(
                    query="8192,16,128",
                    key="8192,16,128",
                    value="8192,16,128",
                    seq_kv_len="2",
                    block="",
                ),
                "Runtime actual_seq_lengths_shape": "",
                "Runtime actual_seq_lengths_values": "",
                "Runtime actual_seq_lengths_kv_shape": "",
                "Runtime actual_seq_lengths_kv_values": "",
                "Runtime avg_seq_len": "",
                "Runtime block_table_valid_blocks": "",
            }
        ]
        jsonl_index = {
            build_jsonl_key(
                {
                    "query_shape": [8192, 16, 128],
                    "key_shape": [8192, 16, 128],
                    "value_shape": [8192, 16, 128],
                    "actual_seq_lengths_kv": [40, 998],
                    "atten_mask_shape": None,
                    "block_table_shape": None,
                }
            ): [
                {
                    "Runtime actual_seq_lengths_shape": "",
                    "Runtime actual_seq_lengths_values": "",
                    "Runtime actual_seq_lengths_kv_shape": "2",
                    "Runtime actual_seq_lengths_kv_values": "40,998",
                    "Runtime avg_seq_len": "519.000000",
                    "Runtime block_table_valid_blocks": "",
                },
                {
                    "Runtime actual_seq_lengths_shape": "",
                    "Runtime actual_seq_lengths_values": "",
                    "Runtime actual_seq_lengths_kv_shape": "2",
                    "Runtime actual_seq_lengths_kv_values": "41,999",
                    "Runtime avg_seq_len": "520.000000",
                    "Runtime block_table_valid_blocks": "",
                },
            ]
        }

        output_rows, matched, total = backfill(
            rows, jsonl_index, "runtime_values_dumped"
        )

        assert matched == 1
        assert total == 2
        assert output_rows[0]["Runtime actual_seq_lengths_kv_values"] == "40,998"
        assert output_rows[0]["Runtime avg_seq_len"] == "519.000000"
        expected_tag = "runtime_values_dumped"
        assert output_rows[0]["Runtime metadata_completeness"] == expected_tag
        assert output_rows[1]["Runtime actual_seq_lengths_kv_values"] == "41,999"
        assert output_rows[1]["Runtime avg_seq_len"] == "520.000000"

    def test_backfill_keeps_unmatched_rows_unchanged(self):
        rows = [
            {
                "Input Shapes": _build_input_shapes(
                    query="8192,16,128",
                    key="8192,16,128",
                    value="8192,16,128",
                    seq_kv_len="2",
                    block="",
                ),
                "Runtime actual_seq_lengths_shape": "",
                "Runtime actual_seq_lengths_values": "",
                "Runtime actual_seq_lengths_kv_shape": "",
                "Runtime actual_seq_lengths_kv_values": "",
                "Runtime avg_seq_len": "",
                "Runtime block_table_valid_blocks": "",
            }
        ]

        output_rows, matched, total = backfill(
            rows, jsonl_index={}, metadata_tag="runtime_values_dumped"
        )

        assert matched == 0
        assert total == 1
        assert output_rows == rows
