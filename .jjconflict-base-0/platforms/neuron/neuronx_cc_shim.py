#!/usr/bin/env python3
"""Python-compatible shim for harness NKI standalone compilation."""

import os
import sys


def main() -> None:
    native = os.path.join(
        os.path.dirname(os.path.realpath(__file__)), "..", "neuronx-cc"
    )
    os.execv(native, [native, *sys.argv[1:]])


if __name__ == "__main__":
    main()
