"""
Compatibility entrypoint delegating to the Local Web Studio (web_app.py).
Also preserves enqueue_output and clean_system_files exports for backward compatibility.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from web_app import clean_system_files, enqueue_output, main  # noqa: F401

if __name__ == "__main__":
    main()
