"""
display_utils.py — Display and formatting utilities for the framework.

Functions:
  - section_header : formatted section header
  - print_dict     : hierarchical dict display
  - progress_bar   : ASCII progress bar
  - quick_summary  : aligned summary line
"""

from __future__ import annotations

from typing import Any, Dict


# =============================================================================
# SECTION HEADERS
# =============================================================================


def section_header(title: str, width: int = 65, char: str = "═") -> None:
    """Print a formatted section header."""
    title_clean = f"  {title}  "
    padding = max(0, width - len(title_clean))
    pad_left = padding // 2
    pad_right = padding - pad_left
    try:
        print("\n" + char * width)
        print(char * pad_left + title_clean + char * pad_right)
        print(char * width)
    except UnicodeEncodeError:
        print("\n" + "=" * width)
        print("=" * pad_left + title_clean + "=" * pad_right)
        print("=" * width)


def subsection(title: str, width: int = 55) -> None:
    """Print a subsection title."""
    print(f"\n  ── {title} {'─' * max(0, width - len(title) - 5)}")


# =============================================================================
# DICT DISPLAY
# =============================================================================


def print_dict(d: Dict[str, Any], indent: int = 2, title: str = "") -> None:
    """Display a dictionary in a hierarchical, readable format."""
    if title:
        print(f"\n{' ' * indent}{title}")
        print(f"{'─' * (indent + len(title))}")
    _print_dict_recursive(d, indent)


def _print_dict_recursive(d: Any, indent: int, level: int = 0) -> None:
    prefix = "  " * (level + indent // 2)
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                print(f"{prefix}{k} :")
                _print_dict_recursive(v, indent, level + 1)
            else:
                v_str = f"{v:.4f}" if isinstance(v, float) else str(v)
                print(f"{prefix}{k:<35} : {v_str}")
    elif isinstance(d, list):
        for item in d[:10]:
            try:
                print(f"{prefix}• {item}")
            except UnicodeEncodeError:
                print(f"{prefix}- {item}")
        if len(d) > 10:
            print(f"{prefix}... ({len(d) - 10} more)")
    else:
        print(f"{prefix}{d}")


# =============================================================================
# PROGRESS BAR
# =============================================================================


def progress_bar(
    current: int,
    total: int,
    prefix: str = "",
    suffix: str = "",
    length: int = 40,
    fill: str = "█",
) -> None:
    """Display an ASCII progress bar."""
    pct = current / total if total > 0 else 0
    filled = int(length * pct)
    bar = fill * filled + "─" * (length - filled)
    print(f"\r  {prefix} |{bar}| {pct*100:.1f}% {suffix}", end="", flush=True)
    if current >= total:
        print()


# =============================================================================
# QUICK SUMMARY
# =============================================================================


def quick_summary(label: str, value: Any, unit: str = "", width: int = 35) -> None:
    """Print an aligned summary line."""
    v_str = f"{value:.4f}" if isinstance(value, float) else str(value)
    print(f"  {label:<{width}} : {v_str} {unit}")
