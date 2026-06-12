"""Shared helpers for writing minimal pytest JUnit XML in regression tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

_JUNIT_XML_HEADER = '<?xml version="1.0" encoding="utf-8"?>'

SAMPLE_MULTI_CASE_JUNIT = """\
<?xml version="1.0" encoding="utf-8"?>
<testsuites>
  <testsuite name="pytest" tests="6" failures="2" errors="1" skipped="0" time="45.3">
    <testcase classname="tests.smoke.test_a" name="test_x" time="1.0"/>
    <testcase classname="tests.smoke.test_b" name="test_y" time="1.0"/>
    <testcase classname="tests.regression.cli.test_run" name="test_z" time="1.0"/>
    <testcase classname="tests.smoke.test_fail" name="test_broken" time="0.5">
      <failure message="AssertionError: 1 != 2">E   AssertionError: 1 != 2</failure>
    </testcase>
    <testcase classname="tests.smoke.test_fail" name="test_other" time="0.5">
      <failure message="ValueError: bad">E   ValueError: bad</failure>
    </testcase>
    <testcase classname="tests.smoke.test_err" name="test_crash" time="0.3">
      <error message="RuntimeError: boom">E   RuntimeError: boom</error>
    </testcase>
  </testsuite>
</testsuites>
"""

TIMED_RANKING_JUNIT = """\
<testsuite>
  <testcase classname="pkg.mod" name="test_fast" time="0.5"/>
  <testcase classname="pkg.mod" name="test_slow" time="9.0"/>
  <testcase classname="pkg.mod" name="test_mid" time="3.0"/>
</testsuite>
"""


def write_junit_xml_content(path: pathlib.Path, content: str) -> pathlib.Path:
    """Write raw JUnit XML content and return the path."""
    path.write_text(content, encoding="utf-8")
    return path


def write_junit_xml(
    path: pathlib.Path,
    *,
    passed: int = 0,
    failed: int = 0,
    duration: float = 0.0,
    pass_classname: str = "tests.smoke.test_a",
    fail_classname: str = "tests.smoke.test_fail",
    file_path: str | None = None,
) -> None:
    """Write a minimal JUnit XML file with generated pass/fail testcases."""
    file_attr = f' file="{file_path}"' if file_path is not None else ""
    lines = [_JUNIT_XML_HEADER, "<testsuites>", "<testsuite>"]
    lines.extend(
        (f'<testcase classname="{pass_classname}" name="test_pass_{index}"{file_attr} time="{duration}"/>')
        for index in range(passed)
    )
    lines.extend(
        (
            f'<testcase classname="{fail_classname}" name="test_fail_{index}"'
            f'{file_attr} time="{duration}">'
            '<failure message="AssertionError: bad">E   AssertionError: bad</failure>'
            "</testcase>"
        )
        for index in range(failed)
    )
    lines.extend(["</testsuite>", "</testsuites>"])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_phase_junit(
    path: pathlib.Path,
    *,
    file_path: str,
    passed: int,
    failed: int = 0,
    duration: float,
) -> None:
    """Write JUnit XML for one nightly pipeline phase with ``file`` attributes."""
    module = file_path.replace("/", ".").removesuffix(".py")
    write_junit_xml(
        path,
        passed=passed,
        failed=failed,
        duration=duration,
        pass_classname=module,
        fail_classname=module,
        file_path=file_path,
    )
