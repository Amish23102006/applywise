"""Web UI tests (skipped if FastAPI is not installed). The model is faked; no key needed."""
import json

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

import applywise as aw  # noqa: E402
import web  # noqa: E402


def test_index_renders():
    r = TestClient(web.app).get("/")
    assert r.status_code == 200 and "ApplyWise" in r.text


def test_analyse_escapes_user_input(monkeypatch):
    extracted = {"company": "<script>x</script>", "role": "Intern", "must_have": ["Python"], "nice_to_have": [],
                 "eligibility": [], "location": "", "work_mode": "remote", "duration": "", "stipend": "", "start": ""}
    responses = [json.dumps(extracted), "I build FastAPI services in Python."]

    class Fake(aw.BharatCodeClient):
        def __init__(self):
            super().__init__(api_key="k", transport=lambda u, p: (200, {"choices": [
                {"message": {"content": responses.pop(0)}, "finish_reason": "stop"}]}))

    monkeypatch.setattr(aw, "BharatCodeClient", Fake)
    r = TestClient(web.app).post("/analyse", data={
        "listing": "Remote Python intern, 2028 batch", "profile": "Python and FastAPI developer\nGraduation 2028"})
    assert r.status_code == 200
    assert "<script>x</script>" not in r.text and "&lt;script&gt;" in r.text
    assert "Fit 100/100" in r.text and "Evidence-only note" in r.text


def test_api_error_is_shown_not_crashed(monkeypatch):
    class Broken(aw.BharatCodeClient):
        def __init__(self):
            super().__init__(api_key="k", transport=lambda u, p: (401, {"error": "bad key"}))

    monkeypatch.setattr(aw, "BharatCodeClient", Broken)
    r = TestClient(web.app).post("/analyse", data={"listing": "x", "profile": "y"})
    assert r.status_code == 200 and "401" in r.text
