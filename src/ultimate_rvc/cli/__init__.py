"""
Package which defines the command-line interface for the Ultimate RVC
project.
"""

from __future__ import annotations

import os

from ultimate_rvc.core.main import initialize

# BookForge fork: initialize() runs first-run prerequisite + sample-model
# downloads and audio-separator setup (the latter pulls deps this inference-only
# env doesn't ship). BookForge pre-stages every required model and runs offline,
# so it sets URVC_SKIP_INIT=1 to skip this entirely. Default behavior is
# unchanged for anyone running the env without that flag.
if os.getenv("URVC_SKIP_INIT", "0") != "1":
    initialize()
