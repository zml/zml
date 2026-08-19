#!/usr/bin/env python3
import sys

from checkpoint_tools import main

if __name__ == "__main__":
    sys.argv.insert(1, "inspect")
    main()
