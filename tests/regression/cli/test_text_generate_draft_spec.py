# Copyright (c) Huawei Technologies Co., Ltd. All rights reserved.
"""CLI regression for text_generate draft-spec (DFlash/DSpark) wiring."""

from __future__ import annotations

from unittest import TestCase

from cli.inference import text_generate as mod


class TestTextGenerateDraftSpecCli(TestCase):
    """RFC G2/G3: Dflash / DSpark CLI wiring for text_generate."""

    def _parse(self, extra: list[str]):
        argv = [
            "--num-queries",
            "1",
            "--query-length",
            "8",
            "Qwen/Qwen3-32B",
            "--device=TEST_DEVICE",
            *extra,
        ]
        return mod.arg_parse(argv)

    def test_defaults_keep_draft_disabled(self):
        args = self._parse([])
        self.assertIsNone(args.speculative_method)
        self.assertEqual(args.num_speculative_tokens, 0)
        self.assertEqual(args.num_mtp_tokens, 0)

    def test_accepts_speculative_method_dflash_and_maps_block(self):
        args = self._parse(
            [
                "--speculative-method=dflash",
                "--num-speculative-tokens=15",
                "--num-draft-layers=2",
            ]
        )
        self.assertEqual(args.speculative_method, "dflash")
        self.assertEqual(args.num_speculative_tokens, 15)
        self.assertEqual(args.draft_block_size, 16)
        self.assertEqual(args.num_draft_layers, 2)

    def test_accepts_speculative_method_dspark_and_maps_block(self):
        args = self._parse(
            [
                "--speculative-method=dspark",
                "--num-speculative-tokens=7",
                "--dspark-markov-rank=128",
                "--dspark-markov-head=gated",
            ]
        )
        self.assertEqual(args.speculative_method, "dspark")
        self.assertEqual(args.num_speculative_tokens, 7)
        self.assertEqual(args.draft_block_size, 8)
        self.assertEqual(args.dspark_markov_rank, 128)
        self.assertEqual(args.dspark_markov_head, "gated")

    def test_builtin_num_speculative_tokens_maps_to_block_eight(self):
        args = self._parse(["--speculative-method=dflash"])
        self.assertEqual(args.num_speculative_tokens, 7)
        self.assertEqual(args.draft_block_size, 8)

    def test_g3_dependent_without_method_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--num-speculative-tokens=15"])

    def test_g3_shared_draft_without_method_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--num-draft-layers=4"])

    def test_g3_markov_requires_dspark_method(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--dspark-markov-rank=128"])

    def test_g2_dflash_and_mtp_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--num-mtp-tokens", "2"])

    def test_g2_dspark_and_mtp_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dspark", "--num-mtp-tokens", "2"])

    def test_dspark_with_explicit_mtp_zero_ok(self):
        args = self._parse(["--speculative-method=dspark", "--num-speculative-tokens=7", "--num-mtp-tokens", "0"])
        self.assertEqual(args.speculative_method, "dspark")
        self.assertEqual(args.num_mtp_tokens, 0)

    def test_text_generate_has_no_acceptance_length_flag(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--acceptance-length=3"])
