import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
@pytest.mark.parametrize("test_file", ["web_app.test.cjs", "cloud_layout.test.mjs"])
def test_browser_javascript(test_file):
    result = subprocess.run(
        ["node", "--test", str(Path(__file__).with_name(test_file))],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
