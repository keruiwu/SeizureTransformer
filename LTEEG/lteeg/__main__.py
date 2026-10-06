import sys

from .cli import main

if __name__ == "__main__":  # required on Windows (spawn) before any worker process starts
    sys.exit(main())
