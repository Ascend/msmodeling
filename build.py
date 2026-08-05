#!/usr/bin/env python3
"""Department unified entry: build wheel or run tests (``python build.py test --suite``)."""

from __future__ import annotations

from scripts.helpers.build.main import main

if __name__ == "__main__":
    raise SystemExit(main())
