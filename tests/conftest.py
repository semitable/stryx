import pytest
import shutil
from pathlib import Path

APP_FILE = Path(__file__).parent / "e2e" / "app.py"

@pytest.fixture
def app_script(tmp_path):
    """Copies e2e/app.py to tmp_path for execution."""
    dest = tmp_path / "app.py"
    shutil.copy(APP_FILE, dest)
    return dest
