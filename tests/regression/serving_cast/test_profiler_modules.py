# Copyright Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
import importlib
import sys
import types
import unittest
from unittest.mock import patch


PROFILER_INTERFACE_MODULE = "serving_cast.profiler.profiler_interface"
PROFILER_STIME_MODULE = "serving_cast.profiler.profiler_stime"
SERVICE_TYPE = "liuren_simulation"
STABLE_FILE_SIZE = 16
START_TIME = 12.5
END_TIME = 15.0
TASK_NAME = "task-7"
MISSING = object()


class RecordingProfilerBase:
    Level = "CLASS_LEVEL"

    def __init__(self, level):
        self.level = level
        self.calls = []

    def metric(self, name, value):
        self.calls.append(("metric", name, value))
        return self

    def event(self, event_name):
        self.calls.append(("event", event_name))
        return ("event", event_name)

    def span_start(self, span_name):
        self.calls.append(("span_start", span_name))
        return ("span_start", span_name)

    def span_end(self):
        self.calls.append(("span_end",))
        return "span_end"

    def add_meta_info(self, key, value):
        self.calls.append(("add_meta_info", key, value))
        return self


class RecordingInitProfiler:
    instances = []

    def __init__(self, level):
        self.level = level
        self.calls = []
        type(self).instances.append(self)

    def add_meta_info(self, key, value):
        self.calls.append(("add_meta_info", key, value))
        return self


def fresh_import(module_name):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def build_fake_ms_service_profiler(*, profiler_cls, level=MISSING, parse_main=None):
    package = types.ModuleType("ms_service_profiler")
    package.__path__ = []
    package.Profiler = profiler_cls
    if level is not MISSING:
        package.Level = level

    parse_module = types.ModuleType("ms_service_profiler.parse")
    parse_module.main = parse_main or (lambda: None)
    package.parse = parse_module
    return package, parse_module


class TestProfilerStime(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop(PROFILER_STIME_MODULE, None)

    def test_import_uses_top_level_level(self):
        level_marker = object()
        package, parse_module = build_fake_ms_service_profiler(
            profiler_cls=RecordingProfilerBase,
            level=level_marker,
        )

        with patch.dict(
            sys.modules,
            {
                "ms_service_profiler": package,
                "ms_service_profiler.parse": parse_module,
            },
            clear=False,
        ):
            module = fresh_import(PROFILER_STIME_MODULE)

        self.assertIs(module.Level, level_marker)
        self.assertTrue(issubclass(module.SimProfiler, RecordingProfilerBase))

    def test_import_falls_back_to_profiler_level(self):
        package, parse_module = build_fake_ms_service_profiler(profiler_cls=RecordingProfilerBase)

        with patch.dict(
            sys.modules,
            {
                "ms_service_profiler": package,
                "ms_service_profiler.parse": parse_module,
            },
            clear=False,
        ):
            module = fresh_import(PROFILER_STIME_MODULE)

        self.assertEqual(module.Level, RecordingProfilerBase.Level)

    def test_import_requires_profiler_package(self):
        original_import = __import__

        def selective_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "ms_service_profiler" or name.startswith("ms_service_profiler."):
                raise ImportError("blocked for test")
            return original_import(name, globals, locals, fromlist, level)

        with patch("builtins.__import__", side_effect=selective_import):
            with self.assertRaisesRegex(ImportError, "Please install ms_service_profiler"):
                fresh_import(PROFILER_STIME_MODULE)

    def test_import_requires_level_on_fallback(self):
        class ProfilerWithoutLevel:
            pass

        package, parse_module = build_fake_ms_service_profiler(profiler_cls=ProfilerWithoutLevel)

        with patch.dict(
            sys.modules,
            {
                "ms_service_profiler": package,
                "ms_service_profiler.parse": parse_module,
            },
            clear=False,
        ):
            with self.assertRaisesRegex(
                ImportError,
                r"ms_service_profiler\.Profiler has no Level; upgrade ms_service_profiler",
            ):
                fresh_import(PROFILER_STIME_MODULE)

    def test_parse_main_func_delegates_to_parse_module(self):
        parse_calls = []

        def fake_parse_main():
            parse_calls.append(list(sys.argv))

        package, parse_module = build_fake_ms_service_profiler(
            profiler_cls=RecordingProfilerBase,
            parse_main=fake_parse_main,
        )

        with patch.dict(
            sys.modules,
            {
                "ms_service_profiler": package,
                "ms_service_profiler.parse": parse_module,
            },
            clear=False,
        ):
            module = fresh_import(PROFILER_STIME_MODULE)
            original_argv = sys.argv[:]
            try:
                sys.argv = ["parse-profiler", "--input-path", "/tmp/demo"]
                module.parse_main_func()
            finally:
                sys.argv = original_argv

        self.assertEqual(parse_calls, [["parse-profiler", "--input-path", "/tmp/demo"]])

    def test_event_records_logical_timestamps_and_pid(self):
        package, parse_module = build_fake_ms_service_profiler(profiler_cls=RecordingProfilerBase)

        with patch.dict(
            sys.modules,
            {
                "ms_service_profiler": package,
                "ms_service_profiler.parse": parse_module,
            },
            clear=False,
        ):
            module = fresh_import(PROFILER_STIME_MODULE)
            profiler = module.SimProfiler("INFO")
            with (
                patch.object(module, "now", side_effect=[START_TIME, END_TIME]),
                patch.object(
                    module,
                    "current_task_name",
                    return_value=TASK_NAME,
                ),
            ):
                result = profiler.event("Decode")

        self.assertEqual(result, ("event", "Decode"))
        self.assertEqual(
            profiler.calls,
            [
                ("metric", "logical_start_time", START_TIME),
                ("metric", "logical_end_time", END_TIME),
                ("metric", "logical_pid", TASK_NAME),
                ("event", "Decode"),
            ],
        )

    def test_span_start_records_start_timestamp_and_pid(self):
        package, parse_module = build_fake_ms_service_profiler(profiler_cls=RecordingProfilerBase)

        with patch.dict(
            sys.modules,
            {
                "ms_service_profiler": package,
                "ms_service_profiler.parse": parse_module,
            },
            clear=False,
        ):
            module = fresh_import(PROFILER_STIME_MODULE)
            profiler = module.SimProfiler("INFO")
            with (
                patch.object(module, "now", return_value=START_TIME),
                patch.object(
                    module,
                    "current_task_name",
                    return_value=TASK_NAME,
                ),
            ):
                result = profiler.span_start("Prefill")

        self.assertEqual(result, ("span_start", "Prefill"))
        self.assertEqual(
            profiler.calls,
            [
                ("metric", "logical_start_time", START_TIME),
                ("metric", "logical_pid", TASK_NAME),
                ("span_start", "Prefill"),
            ],
        )

    def test_span_end_records_end_timestamp_and_pid(self):
        package, parse_module = build_fake_ms_service_profiler(profiler_cls=RecordingProfilerBase)

        with patch.dict(
            sys.modules,
            {
                "ms_service_profiler": package,
                "ms_service_profiler.parse": parse_module,
            },
            clear=False,
        ):
            module = fresh_import(PROFILER_STIME_MODULE)
            profiler = module.SimProfiler("INFO")
            with (
                patch.object(module, "now", return_value=END_TIME),
                patch.object(
                    module,
                    "current_task_name",
                    return_value=TASK_NAME,
                ),
            ):
                result = profiler.span_end()

        self.assertEqual(result, "span_end")
        self.assertEqual(
            profiler.calls,
            [
                ("metric", "logical_end_time", END_TIME),
                ("metric", "logical_pid", TASK_NAME),
                ("span_end",),
            ],
        )


class TestProfilerInterface(unittest.TestCase):
    def tearDown(self):
        sys.modules.pop(PROFILER_INTERFACE_MODULE, None)

    def test_import_without_supported_profiler_disables_profiling(self):
        empty_stime_module = types.ModuleType("serving_cast.profiler.profiler_stime")

        with patch.dict(
            sys.modules,
            {"serving_cast.profiler.profiler_stime": empty_stime_module},
            clear=False,
        ):
            module = fresh_import(PROFILER_INTERFACE_MODULE)

        self.assertFalse(module.is_profiling_ready())
        with self.assertRaisesRegex(ValueError, "profiling is not supported"):
            module.init_profiling()
        with self.assertRaisesRegex(ValueError, "profiling is not supported"):
            module.parse_profiling_results("/tmp/profiler")
        with self.assertRaisesRegex(RuntimeError, "profiling is not supported"):
            module.get_batch_type([])

    def test_supported_import_exposes_helpers_and_initializes_profiler(self):
        RecordingInitProfiler.instances.clear()
        parse_calls = []

        class LevelStub:
            INFO = "INFO"

        fake_stime = types.ModuleType("serving_cast.profiler.profiler_stime")
        fake_stime.Level = LevelStub
        fake_stime.SimProfiler = RecordingInitProfiler
        fake_stime.parse_main_func = lambda: parse_calls.append(list(sys.argv))

        fake_utils = types.ModuleType("serving_cast.profiler.profiler_utils")

        def fake_get_batch_type(payload):
            return ("batch", payload)

        def fake_get_iter_size_info(queue, increase_iter_size):
            return ("iter", queue, increase_iter_size)

        def fake_queue_profiler(before_queue, after_queue, queue_name):
            return ("queue", before_queue, after_queue, queue_name)

        def fake_record_kv_cache_free_blocks(current_event, req_id, num_free_blocks):
            return ("kv", current_event, req_id, num_free_blocks)

        fake_utils.get_batch_type = fake_get_batch_type
        fake_utils.get_iter_size_info = fake_get_iter_size_info
        fake_utils.queue_profiler = fake_queue_profiler
        fake_utils.record_kv_cache_free_blocks = fake_record_kv_cache_free_blocks

        with patch.dict(
            sys.modules,
            {
                "serving_cast.profiler.profiler_stime": fake_stime,
                "serving_cast.profiler.profiler_utils": fake_utils,
            },
            clear=False,
        ):
            module = fresh_import(PROFILER_INTERFACE_MODULE)
            module.init_profiling()

        self.assertTrue(module.is_profiling_ready())
        self.assertIs(module.Level, LevelStub)
        self.assertIs(module.get_batch_type, fake_get_batch_type)
        self.assertIs(module.get_iter_size_info, fake_get_iter_size_info)
        self.assertIs(module.queue_profiler, fake_queue_profiler)
        self.assertIs(module.record_kv_cache_free_blocks, fake_record_kv_cache_free_blocks)
        self.assertEqual(len(RecordingInitProfiler.instances), 1)
        self.assertEqual(RecordingInitProfiler.instances[0].level, LevelStub.INFO)
        self.assertEqual(
            RecordingInitProfiler.instances[0].calls,
            [("add_meta_info", "service_type", SERVICE_TYPE)],
        )
        self.assertEqual(parse_calls, [])

    def test_parse_profiling_results_waits_for_stable_size_before_parsing(self):
        parse_calls = []

        class LevelStub:
            INFO = "INFO"

        fake_stime = types.ModuleType("serving_cast.profiler.profiler_stime")
        fake_stime.Level = LevelStub
        fake_stime.SimProfiler = RecordingInitProfiler
        fake_stime.parse_main_func = lambda: parse_calls.append(list(sys.argv))

        fake_utils = types.ModuleType("serving_cast.profiler.profiler_utils")
        fake_utils.get_batch_type = lambda payload: payload
        fake_utils.get_iter_size_info = lambda queue, increase_iter_size: (queue, increase_iter_size)
        fake_utils.queue_profiler = lambda before_queue, after_queue, queue_name: None
        fake_utils.record_kv_cache_free_blocks = lambda current_event, req_id, num_free_blocks: None

        profile_dir = "/tmp/profiling-run"
        expected_argv = [
            "python -m ms_service_profiler.parse",
            "--input-path",
            profile_dir,
            "--output-path",
            profile_dir + "_parsed_result",
        ]

        with patch.dict(
            sys.modules,
            {
                "serving_cast.profiler.profiler_stime": fake_stime,
                "serving_cast.profiler.profiler_utils": fake_utils,
            },
            clear=False,
        ):
            module = fresh_import(PROFILER_INTERFACE_MODULE)
            original_argv = sys.argv[:]
            try:
                with (
                    patch.object(module.os, "walk", return_value=[(profile_dir, [], ["result.bin"])]),
                    patch.object(
                        module.os.path,
                        "getsize",
                        return_value=STABLE_FILE_SIZE,
                    ),
                    patch.object(module.time, "sleep") as sleep_mock,
                ):
                    module.parse_profiling_results(profile_dir)
            finally:
                sys.argv = original_argv

        self.assertEqual(parse_calls, [expected_argv])
        self.assertEqual(sys.argv, original_argv)
        self.assertEqual(sleep_mock.call_count, 10)
        sleep_mock.assert_called_with(0.1)


if __name__ == "__main__":
    unittest.main()
