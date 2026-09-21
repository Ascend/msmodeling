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

    def test_dspark_cannot_mix_legacy_mtp_zero(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dspark", "--num-speculative-tokens=7", "--num-mtp-tokens", "0"])

    def test_mtp_method_requires_num_speculative_tokens(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=mtp"])

    def test_mtp_cannot_mix_legacy_num_mtp_tokens(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=mtp", "--num-speculative-tokens=2", "--num-mtp-tokens", "2"])

    def test_text_generate_has_no_acceptance_length_flag(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--acceptance-length=3"])

    def test_accepts_speculative_method_mtp(self):
        args = self._parse(["--speculative-method=mtp", "--num-speculative-tokens=2"])
        self.assertEqual(args.speculative_method, "mtp")
        self.assertEqual(args.num_speculative_tokens, 2)
        self.assertEqual(args.draft_block_size, 3)
        self.assertEqual(args.num_mtp_tokens, 2)

    def test_mtp_legacy_and_new_entry_same_decode_config(self):
        legacy = self._parse(["--decode", "--num-mtp-tokens", "2"])
        new = self._parse(["--decode", "--speculative-method=mtp", "--num-speculative-tokens=2"])
        mod.align_decode_query_length(legacy)
        mod.align_decode_query_length(new)
        self.assertEqual(legacy.num_mtp_tokens, new.num_mtp_tokens)
        self.assertEqual(legacy.query_length, new.query_length)
        self.assertEqual(legacy.query_length, 3)
        self.assertEqual(legacy.num_mtp_tokens, 2)

    def test_mtp_method_with_draft_layers_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=mtp", "--num-speculative-tokens=2", "--num-draft-layers=4"])

    def test_explicit_n_zero_with_method_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=dflash", "--num-speculative-tokens=0"])

    def test_explicit_n_zero_with_mtp_method_fails(self):
        with self.assertRaises(SystemExit):
            self._parse(["--speculative-method=mtp", "--num-speculative-tokens=0"])

    # ── Phase (--prefill / --decode) tests ────────────────────────────

    def test_prefill_decode_mutex_rejects_both(self):
        """--prefill and --decode are mutually exclusive."""
        with self.assertRaises(SystemExit):
            self._parse(["--prefill", "--decode"])

    def test_prefill_explicit_flag(self):
        """--prefill sets phase=prefill and does not conflict with other params.

        Note: There is no args.phase field; phase is determined by decode flag.
        When --prefill is passed, decode=False, which means prefill phase.
        """
        args = self._parse(["--prefill"])
        self.assertTrue(args.prefill)
        self.assertFalse(args.decode)
        # Verify prefill phase
        is_prefill_phase = not args.decode
        self.assertTrue(is_prefill_phase)

    def test_decode_explicit_flag_with_mtp_auto_aligns(self):
        """--decode with MTP auto-aligns query_length to N+1."""
        args = self._parse(
            [
                "--decode",
                "--context-length",
                "4096",
                "--speculative-method=mtp",
                "--num-speculative-tokens=3",
            ]
        )
        mod.align_decode_query_length(args)
        self.assertEqual(args.query_length, 4)  # 3 + 1

    def test_decode_without_mtp_keeps_query_length(self):
        """--decode without MTP/Draft keeps query_length unchanged."""
        args = self._parse(["--decode", "--query-length=3"])
        mod.align_decode_query_length(args)
        self.assertEqual(args.query_length, 3)

    def test_default_phase_is_prefill(self):
        """Backward compat: neither --prefill nor --decode means prefill phase.

        Phase determination: decode=False implies prefill phase (see align_decode_query_length).
        Downstream code (e.g., UserInputConfig.decode) uses this flag to determine phase.
        """
        args = self._parse([])
        # Both flags False means prefill phase (backward compatible)
        self.assertFalse(args.prefill)
        self.assertFalse(args.decode)
        # Verify decode flag is the phase indicator
        is_prefill_phase = not args.decode
        self.assertTrue(is_prefill_phase)

    def test_neither_prefill_nor_decode_warns(self):
        """Neither --prefill nor --decode emits an info log but succeeds.

        Note: This info-level log fires on ALL existing CLI calls that omit both
        flags (backward-compatible default path). It's intentionally NOT a warning
        to avoid confusing users into thinking they're doing something wrong.
        """
        import logging

        with self.assertLogs("cli.registry.validators", level=logging.INFO) as cm:
            args = self._parse([])
        # Should still parse successfully
        self.assertFalse(args.prefill)
        self.assertFalse(args.decode)
        # Should have logged an info message
        self.assertTrue(any("prefill" in msg.lower() and "decode" in msg.lower() for msg in cm.output))

    def test_explicitly_provided_tracks_phase_flags(self):
        """Verify _explicitly_provided correctly tracks --prefill/--decode presence.

        This test directly validates the provided set computation that the
        prefillDecodeMutex validator relies on. Since both flags are store_true
        with no cli_off_flag, presence in provided is the sole determinant.
        """
        from cli.registry.argparse_adapter import _explicitly_provided, build_argparser
        from cli.registry.modules import get_spec

        spec = get_spec("text_generate")
        parser = build_argparser(spec)

        # Case 1: --prefill only
        args = parser.parse_args(
            ["--num-queries", "1", "--query-length", "8", "--prefill", "Qwen/Qwen3-32B", "--device=TEST_DEVICE"]
        )
        provided = _explicitly_provided(
            spec,
            parser,
            args,
            ["--num-queries", "1", "--query-length", "8", "--prefill", "Qwen/Qwen3-32B", "--device=TEST_DEVICE"],
        )
        self.assertIn("prefill", provided)
        self.assertNotIn("decode", provided)

        # Case 2: --decode only
        args = parser.parse_args(
            ["--num-queries", "1", "--query-length", "8", "--decode", "Qwen/Qwen3-32B", "--device=TEST_DEVICE"]
        )
        provided = _explicitly_provided(
            spec,
            parser,
            args,
            ["--num-queries", "1", "--query-length", "8", "--decode", "Qwen/Qwen3-32B", "--device=TEST_DEVICE"],
        )
        self.assertIn("decode", provided)
        self.assertNotIn("prefill", provided)

        # Case 3: neither
        args = parser.parse_args(
            ["--num-queries", "1", "--query-length", "8", "Qwen/Qwen3-32B", "--device=TEST_DEVICE"]
        )
        provided = _explicitly_provided(
            spec, parser, args, ["--num-queries", "1", "--query-length", "8", "Qwen/Qwen3-32B", "--device=TEST_DEVICE"]
        )
        self.assertNotIn("prefill", provided)
        self.assertNotIn("decode", provided)

    def test_prefill_decode_mutex_validator_is_registered(self):
        """Verify prefillDecodeMutex validator is registered in text_generate spec.

        This ensures the validator is actually wired up via ValidatorRef and will
        be called during parse_module_args. If this ValidatorRef is accidentally
        removed, this test will catch it.
        """
        from cli.registry.modules import get_spec

        spec = get_spec("text_generate")
        validator_names = [v.name for v in spec.validators]
        self.assertIn("prefillDecodeMutex", validator_names)

        # Verify it has wants_provided=True (needed for the provided set)
        mutex_validator = next(v for v in spec.validators if v.name == "prefillDecodeMutex")
        self.assertTrue(mutex_validator.wants_provided)

    def test_prefill_decode_mutex_catches_web_ui_bypass(self):
        """Validator rejects both=True even if only one is in provided.

        Web UI may have prefill=True as default. If user only toggles decode
        (so provided={'decode'} but params={'prefill': True, 'decode': True}),
        the validator must still reject — otherwise the CLI would see
        --prefill --decode and fail, creating inconsistent behavior.
        """
        from cli.registry.validators import prefill_decode_mutex

        # Scenario: Web UI sends prefill=True (default) + decode=True (touched)
        params = {"prefill": True, "decode": True}
        provided = {"decode"}  # only decode was explicitly touched
        error = prefill_decode_mutex(params, provided)
        self.assertIsNotNone(error)
        self.assertIn("mutually exclusive", error)
