"""Unit tests for the fusion plugin loader.

These tests exercise the loader's file-handling and idempotency logic only;
they do not depend on torch / ModelRunner. Each test writes throwaway plugin
.py files into a temp dir and asserts load_plugin/load_plugin_dir behavior.
"""

import textwrap
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from tensor_cast.plugins import loader
from tensor_cast.plugins.loader import load_plugin, load_plugin_dir


def _write(dir_path, name, body):
    p = Path(dir_path) / name
    p.write_text(textwrap.dedent(body))
    return str(p)


class PluginLoaderTest(unittest.TestCase):
    def setUp(self):
        # Isolate the module-level dedup set between tests.
        self._saved = set(loader._loaded_plugins)
        loader._loaded_plugins.clear()
        self._tmp = TemporaryDirectory()
        self.tmp = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()
        loader._loaded_plugins.clear()
        loader._loaded_plugins.update(self._saved)

    def test_load_valid_plugin_calls_entry(self):
        # A plugin whose register_all_patterns writes a sentinel file proves
        # the entry was invoked.
        sentinel = Path(self.tmp) / "called.txt"
        path = _write(
            self.tmp,
            "ok_plugin.py",
            f"""
            from pathlib import Path
            def register_all_patterns():
                Path(r"{sentinel}").write_text("ok")
            """,
        )
        self.assertTrue(load_plugin(path))
        self.assertTrue(sentinel.is_file())

    def test_idempotent_second_load_skipped(self):
        counter = Path(self.tmp) / "count.txt"
        path = _write(
            self.tmp,
            "counter_plugin.py",
            f"""
            from pathlib import Path
            def register_all_patterns():
                p = Path(r"{counter}")
                n = int(p.read_text()) if p.exists() else 0
                p.write_text(str(n + 1))
            """,
        )
        self.assertTrue(load_plugin(path))
        self.assertFalse(load_plugin(path))  # second call skipped
        self.assertEqual(counter.read_text(), "1")

    def test_missing_entry_skipped(self):
        path = _write(
            self.tmp,
            "no_entry.py",
            """
            # no register_all_patterns defined
            VALUE = 1
            """,
        )
        self.assertFalse(load_plugin(path))
        self.assertNotIn(str(Path(path).resolve()), loader._loaded_plugins)

    def test_syntax_error_skipped_not_raised(self):
        path = _write(
            self.tmp,
            "broken.py",
            """
            def register_all_patterns(:   # syntax error
                pass
            """,
        )
        # Must not raise; returns False.
        self.assertFalse(load_plugin(path))

    def test_entry_raises_skipped_not_marked_loaded(self):
        path = _write(
            self.tmp,
            "raising.py",
            """
            def register_all_patterns():
                raise RuntimeError("boom")
            """,
        )
        self.assertFalse(load_plugin(path))
        self.assertNotIn(str(Path(path).resolve()), loader._loaded_plugins)

    def test_nonexistent_path_skipped(self):
        self.assertFalse(load_plugin(str(Path(self.tmp) / "nope.py")))

    def test_none_path_skipped_not_raised(self):
        # plugin_path=None is the no-plugin baseline (RFC §4.2 / §5.3):
        # must early-return False, never raise on Path(None).
        self.assertFalse(load_plugin(None))
        self.assertEqual(len(loader._loaded_plugins), 0)

    def test_load_dir_loads_all_and_skips_private(self):
        marker = Path(self.tmp) / "marks"
        marker.mkdir()
        for nm in ("a_plugin.py", "b_plugin.py"):
            _write(
                self.tmp,
                nm,
                f"""
                from pathlib import Path
                def register_all_patterns():
                    (Path(r"{marker}") / "{nm}").write_text("x")
                """,
            )
        # Private file should be skipped by load_plugin_dir.
        _write(
            self.tmp,
            "_private.py",
            """
            def register_all_patterns():
                raise AssertionError("private file must not be loaded")
            """,
        )
        loaded = load_plugin_dir(self.tmp)
        self.assertEqual(loaded, 2)
        self.assertTrue((marker / "a_plugin.py").is_file())
        self.assertTrue((marker / "b_plugin.py").is_file())

    def test_load_dir_missing_returns_zero(self):
        self.assertEqual(load_plugin_dir(str(Path(self.tmp) / "absent")), 0)

    def test_list_loaded_plugins_returns_sorted_paths(self):
        from tensor_cast.plugins import list_loaded_plugins

        p = _write(
            self.tmp,
            "list_test.py",
            """
            def register_all_patterns():
                pass
            """,
        )
        load_plugin(p)
        result = list_loaded_plugins()
        self.assertIn(str(Path(p).resolve()), result)
        self.assertEqual(result, sorted(result))


if __name__ == "__main__":
    unittest.main()
