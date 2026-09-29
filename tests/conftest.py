import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        output_dir=tmp_path / "out",
        models_dir=tmp_path / "models",
        enable_mock_model=True,
        default_model="mock",
        log_level="WARNING",
    )
    base.update(overrides)
    return Settings(**base)


@pytest.fixture
def client(tmp_path):
    with TestClient(create_app(make_settings(tmp_path))) as c:
        yield c


@pytest.fixture
def slow_client(tmp_path):
    """Mock backend takes 0.5 s per job and the queue holds one waiting job."""
    s = make_settings(tmp_path, mock_delay_seconds=0.5, max_queue_size=1)
    with TestClient(create_app(s)) as c:
        yield c
