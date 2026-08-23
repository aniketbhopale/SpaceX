"""Backward-compatible entry point; use ``python extract_report.py`` for the CLI."""
from extract_report import main


if __name__ == "__main__":
    raise SystemExit(main())
