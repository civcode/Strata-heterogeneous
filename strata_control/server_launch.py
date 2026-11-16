"""The native run-script server command, shared by setup and mixed-runtime binding."""
import sys
from pathlib import Path


def server_command(root, config, port, open_browser=False):
    """Build the ordinary Strata server invocation; no engine policy here."""
    return [sys.executable, str(Path(root) / "serve" / "server.py"), "--engine", "strata",
            "--config", str(config), "--port", str(port)] + (["--open"] if open_browser else [])
