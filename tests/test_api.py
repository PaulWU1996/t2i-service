import io
import json
import os
import time

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.config import Settings
from app.main import create_app

from .conftest import make_settings


def wait_done(client, job_id, timeout=10):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/v1/generations/{job_id}").json()
        if body["status"] in ("succeeded", "failed"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


# ------------------------------------------------------------------ basics

def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_no_credentials_needed(client):
    """The orchestrator calls the service directly: no auth headers anywhere."""
    assert "authorization" not in {k.lower() for k in client.headers}
    assert client.get("/v1/models").status_code == 200
    r = client.post("/v1/generations?wait=true", json={"prompt": "no key"})
    assert r.status_code == 200 and r.json()["status"] == "succeeded"


def test_startup_refuses_unknown_default_model(tmp_path):
    with pytest.raises(RuntimeError, match="not one of"):
        with TestClient(create_app(make_settings(tmp_path, default_model="nope"))):
            pass


def test_models_listing(client):
    models = client.get("/v1/models").json()
    assert [m["name"] for m in models] == ["flux2-klein-4b", "mock"]
    assert [m["default"] for m in models] == [False, True]


# ------------------------------------------------------------------ happy paths

def test_generate_wait_true_returns_finished_job_and_png(client):
    r = client.post("/v1/generations?wait=true", json={"prompt": "a red fox", "seed": 42})
    assert r.status_code == 200
    job = r.json()
    assert job["status"] == "succeeded"
    assert job["params"]["seed"] == 42
    assert job["params"]["width"] == 512  # mock default
    assert job["image_url"] == f"/v1/generations/{job['id']}/image"
    assert job["timings"]["generate_seconds"] is not None

    img = client.get(job["image_url"])
    assert img.status_code == 200
    assert img.headers["content-type"] == "image/png"
    assert img.content[:8] == b"\x89PNG\r\n\x1a\n"
    pil = Image.open(io.BytesIO(img.content))
    assert pil.size == (512, 512)
    embedded = json.loads(pil.text["t2i"])  # parameters embedded for reproducibility
    assert embedded["prompt"] == "a red fox"
    assert embedded["params"]["seed"] == 42
    assert embedded["id"] == job["id"]


def test_async_flow_202_then_poll(client):
    r = client.post("/v1/generations", json={"prompt": "async please", "width": 256, "height": 320})
    assert r.status_code in (200, 202)
    job = wait_done(client, r.json()["id"])
    assert job["status"] == "succeeded"
    pil = Image.open(io.BytesIO(client.get(job["image_url"]).content))
    assert pil.size == (256, 320)


def test_random_seed_is_recorded_when_omitted(client):
    a = client.post("/v1/generations?wait=true", json={"prompt": "p"}).json()
    b = client.post("/v1/generations?wait=true", json={"prompt": "p"}).json()
    assert isinstance(a["params"]["seed"], int)
    assert a["params"]["seed"] != b["params"]["seed"]


def test_non_ascii_prompt_roundtrip(client):
    prompt = "北京的秋天，金色的银杏叶 🍂"
    job = client.post("/v1/generations?wait=true", json={"prompt": prompt}).json()
    assert job["prompt"] == prompt
    pil = Image.open(io.BytesIO(client.get(job["image_url"]).content))
    assert json.loads(pil.text["t2i"])["prompt"] == prompt


def test_zero_guidance_is_respected(client):
    job = client.post("/v1/generations?wait=true", json={"prompt": "p", "guidance": 0}).json()
    assert job["params"]["guidance"] == 0.0


# ------------------------------------------------------------------ validation

@pytest.mark.parametrize("body", [
    {"prompt": ""},
    {"prompt": "   "},
    {},
    {"prompt": "x", "width": 500},            # not a multiple of 16
    {"prompt": "x", "width": 128},            # below minimum
    {"prompt": "x", "steps": 0},
    {"prompt": "x", "seed": -1},
    {"prompt": "x", "image": "aGk="},         # img2img not supported yet: rejected, not ignored
    {"prompt": "x", "negative_prompt": "y"},  # unknown field
])
def test_invalid_requests_are_422(client, body):
    assert client.post("/v1/generations", json=body).status_code == 422


def test_too_large_image_422(client):
    r = client.post("/v1/generations", json={"prompt": "x", "width": 4096, "height": 4096})
    assert r.status_code == 422
    r = client.post("/v1/generations", json={"prompt": "x", "width": 2048, "height": 2048})
    assert r.status_code == 422  # within max_side but over max_pixels


def test_prompt_too_long_422(tmp_path):
    with TestClient(create_app(make_settings(tmp_path, max_prompt_chars=10))) as c:
        assert c.post("/v1/generations", json={"prompt": "x" * 11}).status_code == 422


def test_unknown_model_400(client):
    r = client.post("/v1/generations", json={"prompt": "x", "model": "does-not-exist"})
    assert r.status_code == 400
    assert "mock" in r.json()["detail"]


# ------------------------------------------------------------------ lookup / failure / queue

def test_unknown_and_malformed_ids_404(client):
    assert client.get("/v1/generations/" + "0" * 32).status_code == 404
    assert client.get("/v1/generations/not-an-id").status_code == 404
    assert client.get("/v1/generations/..%2F..%2Fetc%2Fpasswd/image").status_code == 404


def test_failed_job_reports_error_and_no_image(client):
    job = client.post("/v1/generations?wait=true", json={"prompt": "please FAIL"}).json()
    assert job["status"] == "failed"
    assert "mock backend forced failure" in job["error"]
    assert job["image_url"] is None
    assert client.get(f"/v1/generations/{job['id']}/image").status_code == 404
    # and the worker survives to serve the next job
    ok = client.post("/v1/generations?wait=true", json={"prompt": "fine"}).json()
    assert ok["status"] == "succeeded"


def test_wait_timeout_returns_202_with_unfinished_job(slow_client):
    r = slow_client.post("/v1/generations?wait=true&timeout=0.1", json={"prompt": "slow"})
    assert r.status_code == 202
    assert r.json()["status"] in ("queued", "running")
    # image not ready yet -> 409
    assert slow_client.get(f"/v1/generations/{r.json()['id']}/image").status_code == 409
    assert wait_done(slow_client, r.json()["id"])["status"] == "succeeded"


def test_queue_full_returns_429(slow_client):
    codes = [slow_client.post("/v1/generations", json={"prompt": f"j{i}"}).status_code for i in range(6)]
    assert 429 in codes
    assert any(c in (200, 202) for c in codes)
    r = slow_client.post("/v1/generations", json={"prompt": "again"})
    if r.status_code == 429:
        assert r.headers["retry-after"] == "10"


def test_results_survive_store_loss(client):
    """Simulates a service restart: in-memory store empty, results still on disk."""
    job = client.post("/v1/generations?wait=true", json={"prompt": "persist me", "seed": 7}).json()
    client.app.state.store._jobs.clear()
    again = client.get(f"/v1/generations/{job['id']}")
    assert again.status_code == 200
    assert again.json()["params"]["seed"] == 7
    assert client.get(again.json()["image_url"]).status_code == 200


def test_retention_sweep(client):
    job = client.post("/v1/generations?wait=true", json={"prompt": "old"}).json()
    storage = client.app.state.storage
    old = time.time() - 40 * 86400
    for p in (storage.image_path(job["id"]), storage.meta_path(job["id"])):
        os.utime(p, (old, old))
    fresh = client.post("/v1/generations?wait=true", json={"prompt": "new"}).json()
    assert storage.sweep(30) == 2
    client.app.state.store._jobs.clear()
    assert client.get(f"/v1/generations/{job['id']}").status_code == 404
    assert client.get(f"/v1/generations/{fresh['id']}").status_code == 200
    assert storage.sweep(0) == 0  # 0 disables cleanup
