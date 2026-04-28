from unittest import TestCase

from cli.utils import get_common_argparser


class TestCliUtils(TestCase):
    def test_common_argparser_reserved_memory_default_is_zero(self):
        parser = get_common_argparser()

        args = parser.parse_args(["Qwen/Qwen3-32B"])

        self.assertEqual(args.reserved_memory_gb, 0.0)

    def test_common_argparser_reserved_memory_default_can_be_overridden(self):
        parser = get_common_argparser(reserved_memory_gb_default=10.0)

        args = parser.parse_args(["Qwen/Qwen3-32B"])

        self.assertEqual(args.reserved_memory_gb, 10.0)
