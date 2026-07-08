#!/usr/bin/env python3
"""Generate Zig terminal color constants from named RGB hex colors."""

from __future__ import annotations

import re
import sys


ANSI16 = [
    (0x00, 0x00, 0x00),
    (0x80, 0x00, 0x00),
    (0x00, 0x80, 0x00),
    (0x80, 0x80, 0x00),
    (0x00, 0x00, 0x80),
    (0x80, 0x00, 0x80),
    (0x00, 0x80, 0x80),
    (0xC0, 0xC0, 0xC0),
    (0x80, 0x80, 0x80),
    (0xFF, 0x00, 0x00),
    (0x00, 0xFF, 0x00),
    (0xFF, 0xFF, 0x00),
    (0x00, 0x00, 0xFF),
    (0xFF, 0x00, 0xFF),
    (0x00, 0xFF, 0xFF),
    (0xFF, 0xFF, 0xFF),
]
CUBE_LEVELS = [0, 95, 135, 175, 215, 255]
HEX_RE = re.compile(r"^(?:#|0x)?([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$")
NAME_RE = re.compile(r"[^0-9A-Za-z]+")
ESCAPE_TYPES = (
    "xterm-fg",
    "xterm-bg",
    "true-fg",
    "true-bg",
)


def xterm_palette() -> list[tuple[int, int, int]]:
    palette = ANSI16[:]
    for r in CUBE_LEVELS:
        for g in CUBE_LEVELS:
            for b in CUBE_LEVELS:
                palette.append((r, g, b))
    for i in range(24):
        level = 8 + (10 * i)
        palette.append((level, level, level))
    return palette


PALETTE = xterm_palette()


def parse_rgb(value: str) -> tuple[int, int, int]:
    match = HEX_RE.match(value.strip())
    if not match:
        raise ValueError("expected #rgb, rgb, #rrggbb, rrggbb, 0xrgb, or 0xrrggbb")

    hex_value = match.group(1)
    if len(hex_value) == 3:
        hex_value = "".join(ch * 2 for ch in hex_value)

    return tuple(int(hex_value[i : i + 2], 16) for i in (0, 2, 4))


def parse_named_rgb(value: str) -> tuple[str, tuple[int, int, int]]:
    if "=" not in value:
        raise ValueError("expected name=rgb")

    name, rgb_text = value.split("=", 1)
    name = const_name(name)
    if not name:
        raise ValueError("expected non-empty color name before =")

    return name, parse_rgb(rgb_text)


def const_name(value: str) -> str:
    name = NAME_RE.sub("_", value.strip()).strip("_").upper()
    if name and name[0].isdigit():
        name = f"COLOR_{name}"
    return name


def nearest_xterm(rgb: tuple[int, int, int]) -> int:
    red, green, blue = rgb
    return min(
        range(len(PALETTE)),
        key=lambda index: sum(
            (component - target) ** 2
            for component, target in zip((red, green, blue), PALETTE[index])
        ),
    )


def escape_codes(rgb: tuple[int, int, int]) -> dict[str, str]:
    red, green, blue = rgb
    index = nearest_xterm(rgb)
    return {
        "xterm-fg": f"\\x1b[38;5;{index}m",
        "xterm-bg": f"\\x1b[48;5;{index}m",
        "true-fg": f"\\x1b[38;2;{red};{green};{blue}m",
        "true-bg": f"\\x1b[48;2;{red};{green};{blue}m",
    }


def main(argv: list[str]) -> int:
    if not argv:
        print(f"usage: {sys.argv[0]} NAME=COLOR [NAME=COLOR ...]", file=sys.stderr)
        print("example: tools/rgb_to_xterm.py cyan=#00ffff logo_blue=0080ff", file=sys.stderr)
        return 2

    colors: list[tuple[str, dict[str, str]]] = []
    names: set[str] = set()
    errors = 0
    for arg in argv:
        try:
            name, rgb = parse_named_rgb(arg)
        except ValueError as err:
            print(f"{arg}: {err}", file=sys.stderr)
            errors = 1
            continue

        if name in names:
            print(f"{arg}: duplicate normalized color name {name}", file=sys.stderr)
            errors = 1
            continue

        names.add(name)
        colors.append((name, escape_codes(rgb)))

    if colors:
        for index, escape_type in enumerate(ESCAPE_TYPES):
            if index:
                print()
            print(f"# {escape_type}")
            for name, escapes in colors:
                print(f'const {name}="{escapes[escape_type]}";')

    return errors


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
