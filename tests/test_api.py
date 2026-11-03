"""End-to-end smoke tests on images that hold no team: the models load and run, and the HTTP
API answers. (Accuracy is measured with `recognizer eval` on labeled screenshots.)"""

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from recognizer.pipeline import Recognizer


def png(width: int = 640, height: int = 360) -> bytes:
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("RECOGNIZER_TOKEN", raising=False)
    return TestClient(server.app)


def test_blank_image_has_no_teams():
    assert Recognizer(read_names=False).recognize(np.zeros((720, 1280, 3), np.uint8)) == []


def test_health(client):
    assert client.get("/health").json() == {"ok": True}


def test_recognize_raw_bytes(client):
    response = client.post("/recognize", content=png())
    assert response.status_code == 200
    assert response.json()["teams"] == []


def test_recognize_multipart(client):
    response = client.post("/recognize", files={"image": ("shot.png", png(), "image/png")})
    assert response.status_code == 200
    assert response.json()["teams"] == []


def test_rejects_non_images(client):
    assert client.post("/recognize", content=b"not an image").status_code == 400


def test_token(client, monkeypatch):
    monkeypatch.setenv("RECOGNIZER_TOKEN", "test-token")
    assert client.post("/recognize", content=png()).status_code == 401
    ok = client.post("/recognize", content=png(), headers={"Authorization": "Bearer test-token"})
    assert ok.status_code == 200
