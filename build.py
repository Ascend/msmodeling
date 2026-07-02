#!/usr/bin/env python3
"""Department unified entry: build wheel or run CI gate."""

from __future__ import annotations

from scripts.helpers.build.main import main

if __name__ == "__main__":
    raise SystemExit(main())
