from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def alice() -> str:
    return (ROOT / "examples" / "alice.txt").read_text(encoding="utf-8")
