import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_browser_api_and_terminal_lifecycle():
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name("web_app.test.cjs"))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
