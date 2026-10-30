from __future__ import annotations

from pathlib import Path

import pytest

SAMPLES = Path(__file__).resolve().parents[2] / "samples"


@pytest.fixture
def sample_paths() -> list[Path]:
    paths = sorted(SAMPLES.glob("*.vtt"))
    assert paths, "sample transcripts are missing"
    return paths
