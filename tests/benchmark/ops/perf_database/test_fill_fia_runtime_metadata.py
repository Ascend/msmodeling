"""Tests for fill_fia_runtime_metadata JOIN-based backfill."""

# pylint: disable=no-name-in-module
import json
import tempfile
from pathlib import Path

from tools.perf_data_collection.fill_fia_runtime_metadata import (
    backfill,
    build_csv_key,
    build_jsonl_key,
    build_runtime_payload,
    load_jsonl,
)


class TestBuildCsvKey:
    """Extract JOIN key from CSV Input Shapes 31-slot format."""

    def test_mla_prefill_tnd_3d(self):
        """MLA prefill: 3D TND shapes, mask present, no block_table."""
        row = {
            "Input Shapes": (
                '"8192,16,128;8192,16,128;8192,16,128;;2048,2048;2;2;;;;;;;;;;;;;;;;;;8192,16,64;8192,16,64;;;;;"'
            ),
        }
        key = build_csv_key(row)
        assert key == (
            (8192, 16, 128),  # Q
            (8192, 16, 128),  # K
            (8192, 16, 128),  # V
            2,  # num_seqs_kv (slot[6])
            (2048, 2048),  # atten_mask
            None,  # block_table
        )

    def test_mla_prefill_3seq(self):
        """3-seq batch: slot[6] = 3."""
        row = {
            "Input Shapes": (
                '"8192,16,128;8192,16,128;8192,16,128;;2048,2048;3;3;;;;;;;;;;;;;;;;;;8192,16,64;8192,16,64;;;;;"'
            ),
        }
        key = build_csv_key(row)
        assert key[3] == 3  # num_seqs_kv

    def test_mla_chunk_no_mask(self):
        """Chunk context: Q != K, no atten_mask."""
        row = {
            "Input Shapes": ('"4105,16,128;4093,16,128;4093,16,128;;;2;2;;;;;;;;;;;;;;;;;;4105,16,64;4093,16,64;;;;;"'),
        }
        key = build_csv_key(row)
        assert key[0] == (4105, 16, 128)  # Q
        assert key[1] == (4093, 16, 128)  # K != Q
        assert key[4] is None  # no mask

    def test_mla_decode_4d_bnsd(self):
        """MLA decode: 4D BNSD, block_table present."""
        row = {
            "Input Shapes": (
                '"1,16,1,512;1170,1,128,512;1170,1,128,512;;;;1;;;;;;;;1,512;;;;;;;;;1,16,1,64;1170,1,128,64;;;;;"'
            ),
        }
        key = build_csv_key(row)
        assert key[0] == (1, 16, 1, 512)  # Q 4D
        assert key[5] == (1, 512)  # block_table

    def test_empty_slots_produce_none(self):
        """Empty or missing slots should produce None, not crash."""
        row = {"Input Shapes": '"4099,16,128;4099,16,128;4099,16,128"'}
        key = build_csv_key(row)
        assert key[3] is None  # num_seqs_kv
        assert key[4] is None  # mask
        assert key[5] is None  # block_table


class TestBuildJsonlKey:
    """Extract JOIN key from JSONL dump record."""

    def test_mla_prefill(self):
        record = {
            "query_shape": [8192, 16, 128],
            "key_shape": [8192, 16, 128],
            "value_shape": [8192, 16, 128],
            "actual_seq_lengths_kv": [4099, 8192],
            "atten_mask_shape": [2048, 2048],
            "block_table_shape": None,
        }
        key = build_jsonl_key(record)
        assert key == (
            (8192, 16, 128),
            (8192, 16, 128),
            (8192, 16, 128),
            2,  # len([4099, 8192])
            (2048, 2048),
            None,
        )

    def test_mla_decode(self):
        record = {
            "query_shape": [1, 16, 1, 512],
            "key_shape": [1170, 1, 128, 512],
            "value_shape": [1170, 1, 128, 512],
            "actual_seq_lengths_kv": [1],
            "atten_mask_shape": None,
            "block_table_shape": [1, 512],
        }
        key = build_jsonl_key(record)
        assert key[3] == 1  # num_seqs_kv
        assert key[5] == (1, 512)  # block_table

    def test_null_seq_lengths_kv(self):
        """actual_seq_lengths_kv can be None (some decode paths)."""
        record = {
            "query_shape": [1, 16, 1, 512],
            "key_shape": [1170, 1, 128, 512],
            "value_shape": [1170, 1, 128, 512],
            "actual_seq_lengths_kv": None,
            "atten_mask_shape": None,
            "block_table_shape": [1, 512],
        }
        key = build_jsonl_key(record)
        assert key[3] is None  # num_seqs_kv = None


class TestBuildRuntimePayload:
    def test_prefill_payload(self):
        record = {
            "actual_seq_lengths": [4099, 8192],
            "actual_seq_lengths_kv": [4099, 8192],
            "block_table_valid_blocks": None,
            "num_heads": 16,
            "num_key_value_heads": 16,
            "sparse_mode": 3,
            "input_layout": "TND",
            "block_size": 0,
        }
        payload = build_runtime_payload(record)
        assert payload["Runtime actual_seq_lengths_values"] == "4099,8192"
        assert payload["Runtime actual_seq_lengths_kv_values"] == "4099,8192"
        assert payload["Runtime avg_seq_len"] == "6145.500000"
        assert payload["Runtime num_heads"] == "16"
        assert payload["Runtime sparse_mode"] == "3"

    def test_none_seq_lengths(self):
        record = {
            "actual_seq_lengths": None,
            "actual_seq_lengths_kv": [1],
            "block_table_valid_blocks": [1],
            "num_heads": 16,
            "num_key_value_heads": 1,
            "sparse_mode": 0,
            "input_layout": "BNSD_NBSD",
            "block_size": 128,
        }
        payload = build_runtime_payload(record)
        assert payload["Runtime actual_seq_lengths_values"] == ""
        assert payload["Runtime actual_seq_lengths_kv_values"] == "1"
        assert payload["Runtime avg_seq_len"] == "1.000000"


class TestBackfill:
    """End-to-end backfill with in-memory data."""

    def _make_csv_row(self, input_shapes: str, duration: str = "100.0") -> dict:
        return {
            "Input Shapes": input_shapes,
            "Profiling Average Duration(us)": duration,
        }

    def _make_jsonl_record(
        self,
        q,
        k,
        v,
        seq_kv,
        mask=None,
        bt=None,
        sparse=3,
        num_heads=16,
        kv_heads=16,
    ) -> dict:
        return {
            "query_shape": q,
            "key_shape": k,
            "value_shape": v,
            "actual_seq_lengths": seq_kv,  # simplified: same as kv for prefill
            "actual_seq_lengths_kv": seq_kv,
            "atten_mask_shape": mask,
            "block_table_shape": bt,
            "block_table_valid_blocks": None,
            "num_heads": num_heads,
            "num_key_value_heads": kv_heads,
            "sparse_mode": sparse,
            "input_layout": "TND",
            "block_size": 0,
        }

    def test_1_to_1_match(self):
        """One CSV row matches exactly one JSONL record."""
        csv_row = self._make_csv_row(
            '"4099,16,128;4099,16,128;4099,16,128;;2048,2048;1;1;;;;;;;;;;;;;;;;;;4099,16,64;4099,16,64;;;;;"',
            "600.0",
        )
        jsonl_record = self._make_jsonl_record(
            [4099, 16, 128],
            [4099, 16, 128],
            [4099, 16, 128],
            [4099],
            mask=[2048, 2048],
        )
        jsonl_index = {build_jsonl_key(jsonl_record): [build_runtime_payload(jsonl_record)]}
        rows, matched, total = backfill([csv_row], jsonl_index, "test")

        assert matched == 1
        assert total == 1
        assert rows[0]["Profiling Average Duration(us)"] == "600.0"
        assert rows[0]["Runtime avg_seq_len"] == "4099.000000"

    def test_1_to_n_expansion(self):
        """One CSV row matches two JSONL records with different seq values."""
        csv_row = self._make_csv_row(
            '"8192,16,128;8192,16,128;8192,16,128;;2048,2048;;2;;;;;;;;;;;;;;;;;;8192,16,64;8192,16,64;;;;;"',
            "1200.0",
        )
        rec_a = self._make_jsonl_record(
            [8192, 16, 128],
            [8192, 16, 128],
            [8192, 16, 128],
            [4099, 8192],
            mask=[2048, 2048],
        )
        rec_b = self._make_jsonl_record(
            [8192, 16, 128],
            [8192, 16, 128],
            [8192, 16, 128],
            [100, 8192],
            mask=[2048, 2048],
        )
        key = build_jsonl_key(rec_a)
        jsonl_index = {key: [build_runtime_payload(rec_a), build_runtime_payload(rec_b)]}
        rows, matched, total = backfill([csv_row], jsonl_index, "test")

        assert matched == 1
        assert total == 2
        assert rows[0]["Profiling Average Duration(us)"] == "1200.0"
        assert rows[1]["Profiling Average Duration(us)"] == "1200.0"
        avg_values = {r["Runtime avg_seq_len"] for r in rows}
        assert len(avg_values) == 2  # two different avg_seq_len

    def test_no_match_passthrough(self):
        """Unmatched CSV row passes through unchanged."""
        csv_row = self._make_csv_row('"999,16,128;999,16,128;999,16,128"', "50.0")
        rows, matched, total = backfill([csv_row], {}, "test")

        assert matched == 0
        assert total == 1
        assert rows[0]["Profiling Average Duration(us)"] == "50.0"
        assert rows[0].get("Runtime avg_seq_len") is None

    def test_prefill_vs_chunk_no_cross_match(self):
        """Prefill (K==Q, mask) and chunk (K!=Q, no mask) don't cross-match."""
        prefill_csv = self._make_csv_row(
            '"4105,16,128;4105,16,128;4105,16,128;;2048,2048;;2;;;;;;;;;;;;;;;;;;4105,16,64;4105,16,64;;;;;"',
        )
        chunk_csv = self._make_csv_row(
            '"4105,16,128;4093,16,128;4093,16,128;;;;2;;;;;;;;;;;;;;;;;;4105,16,64;4093,16,64;;;;;"',
        )
        prefill_rec = self._make_jsonl_record(
            [4105, 16, 128],
            [4105, 16, 128],
            [4105, 16, 128],
            [6, 4105],
            mask=[2048, 2048],
        )
        chunk_rec = self._make_jsonl_record(
            [4105, 16, 128],
            [4093, 16, 128],
            [4093, 16, 128],
            [4093, 4093],
            mask=None,
            sparse=0,
        )
        jsonl_index = {
            build_jsonl_key(prefill_rec): [build_runtime_payload(prefill_rec)],
            build_jsonl_key(chunk_rec): [build_runtime_payload(chunk_rec)],
        }
        rows, matched, total = backfill([prefill_csv, chunk_csv], jsonl_index, "test")

        assert matched == 2
        assert total == 2
        # Prefill row got prefill runtime
        assert rows[0]["Runtime sparse_mode"] == "3"
        assert rows[0]["Runtime actual_seq_lengths_kv_values"] == "6,4105"
        # Chunk row got chunk runtime
        assert rows[1]["Runtime sparse_mode"] == "0"
        assert rows[1]["Runtime actual_seq_lengths_kv_values"] == "4093,4093"

    def test_decode_seq_q_differs_from_seq_kv(self):
        """Decode: actual_seq_lengths (Q=1 per seq) != actual_seq_lengths_kv (KV=4500)."""
        csv_row = self._make_csv_row(
            '"1,16,1,512;1170,1,128,512;1170,1,128,512;;;;1;;;;;;;;1,512;;;;;;;;;;1,16,1,64;1170,1,128,64;;;;;"',
        )
        rec = {
            "query_shape": [1, 16, 1, 512],
            "key_shape": [1170, 1, 128, 512],
            "value_shape": [1170, 1, 128, 512],
            "actual_seq_lengths": None,
            "actual_seq_lengths_kv": [4500],
            "atten_mask_shape": None,
            "block_table_shape": [1, 512],
            "block_table_valid_blocks": [36],
            "num_heads": 16,
            "num_key_value_heads": 1,
            "sparse_mode": 0,
            "input_layout": "BNSD_NBSD",
            "block_size": 128,
        }
        jsonl_index = {build_jsonl_key(rec): [build_runtime_payload(rec)]}
        rows, matched, total = backfill([csv_row], jsonl_index, "test")

        assert matched == 1
        assert total == 1
        assert rows[0]["Runtime actual_seq_lengths_values"] == ""  # None → empty
        assert rows[0]["Runtime actual_seq_lengths_kv_values"] == "4500"
        assert rows[0]["Runtime avg_seq_len"] == "4500.000000"
        assert rows[0]["Runtime num_key_value_heads"] == "1"
        assert rows[0]["Runtime block_table_valid_blocks"] == "36"


class TestLoadJsonl:
    def test_dedup_identical_records(self):
        """Duplicate JSONL lines with same seq values produce one payload."""
        records = [
            {
                "query_shape": [100, 16, 128],
                "key_shape": [100, 16, 128],
                "value_shape": [100, 16, 128],
                "actual_seq_lengths": [100],
                "actual_seq_lengths_kv": [100],
                "atten_mask_shape": None,
                "block_table_shape": None,
                "block_table_valid_blocks": None,
                "num_heads": 16,
                "num_key_value_heads": 16,
                "sparse_mode": 3,
                "input_layout": "TND",
                "block_size": 0,
            },
        ] * 5  # 5 identical records

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for r in records:
                f.write(json.dumps(r) + "\n")
            path = Path(f.name)

        index = load_jsonl(path)
        path.unlink()

        assert len(index) == 1  # one unique key
        payloads = next(iter(index.values()))
        assert len(payloads) == 1  # deduped to one payload
