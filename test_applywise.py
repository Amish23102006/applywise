"""Offline tests: the BharatCode API is replaced with a fake transport, so no key or network is needed."""
import json
from datetime import date
from pathlib import Path

import pytest

import applywise as aw

HERE = Path(__file__).parent
PROFILE = aw.load_profile((HERE / "sample_profile.md").read_text(), location="Indore",
                          open_to=["remote"], available_from="April 2027")
LISTING = (HERE / "sample_listing.txt").read_text()
NOV_2026 = date(2026, 11, 9)


def fake_transport(responses):
    """Returns queued (status, body) pairs and records the payloads it received."""
    calls = []

    def _t(url, payload):
        calls.append((url, payload))
        status, content, finish = responses.pop(0)
        body = {"choices": [{"message": {"content": content}, "finish_reason": finish}]} if status == 200 else {"error": "x"}
        return status, body

    _t.calls = calls
    return _t


EXTRACTED = json.dumps({
    "company": "Acme Robotics", "role": "AI Engineer Intern",
    "must_have": ["Python", "REST APIs", "LLMs", "Git", "RAG"],
    "nice_to_have": ["Kubernetes", "TypeScript", "computer vision"],
    "location": "Bengaluru", "work_mode": "onsite", "duration": "6 months",
    "stipend": "INR 30,000/month", "start": "immediate",
    "eligibility": ["open to 2026 and 2027 graduates only"],
})


# ---------------- eligibility gate ----------------
def test_grad_year_autodetected():
    assert PROFILE.grad_year == 2028


def test_cohort_mismatch_is_not_eligible():
    g = aw.eligibility_gate(LISTING, PROFILE, today=NOV_2026)
    assert g.verdict == "NOT_ELIGIBLE"
    assert any("2026" in r and "2028" in r for r in g.reasons)


def test_cohort_match_is_fine():
    g = aw.eligibility_gate("Open to the 2028 batch. Remote.", PROFILE, today=NOV_2026)
    assert g.verdict == "ELIGIBLE"


@pytest.mark.parametrize("text", ["2026/27 graduates", "graduates from the 2026 batch", "2026 pass-outs only"])
def test_cohort_patterns(text):
    assert aw.eligibility_gate(text, PROFILE, today=NOV_2026).verdict == "NOT_ELIGIBLE"


def test_final_year_rule():
    g = aw.eligibility_gate("Eligibility: final year (8th semester) students only.", PROFILE, today=NOV_2026)
    assert g.verdict == "NOT_ELIGIBLE"
    assert "year 3" in " ".join(g.reasons)


def test_final_year_ok_when_in_final_year():
    g = aw.eligibility_gate("Final year students only.", PROFILE, today=date(2027, 9, 1))
    assert g.verdict == "ELIGIBLE"


def test_any_year_overrides_final_year_mention():
    g = aw.eligibility_gate("College students from any year, including final year.", PROFILE, today=NOV_2026)
    assert g.verdict == "ELIGIBLE"


def test_onsite_and_immediate_start_need_checking():
    g = aw.eligibility_gate("Work from office in Pune. Immediate joiners.", PROFILE, today=NOV_2026)
    assert g.verdict == "CHECK"
    assert any("onsite" in r for r in g.reasons) and any("April 2027" in r for r in g.reasons)


def test_remote_role_in_any_city_is_fine():
    assert aw.eligibility_gate("Remote, work from home. 3 months.", PROFILE, today=NOV_2026).verdict == "ELIGIBLE"


def test_fee_is_a_red_flag():
    g = aw.eligibility_gate("Remote internship. Pay a registration fee of INR 999.", PROFILE, today=NOV_2026)
    assert g.verdict == "NOT_ELIGIBLE"


def test_unpaid_needs_checking():
    assert aw.eligibility_gate("Remote. Unpaid internship.", PROFILE, today=NOV_2026).verdict == "CHECK"


# ---------------- evidence matching ----------------
def test_evidence_found_and_missing():
    ev = aw.match_evidence(["Python", "RAG", "Kubernetes"], PROFILE, "must")
    got = {e.requirement: e.met for e in ev}
    assert got == {"Python": True, "RAG": False, "Kubernetes": True}


def test_short_terms_match_whole_words_only():
    p = aw.load_profile("I like programming in Go and Rust.\nGraduation 2028")
    assert aw.match_evidence(["Go"], p, "must")[0].met
    p2 = aw.load_profile("Worked on Google products and algorithms.\nGraduation 2028")
    assert not aw.match_evidence(["Go"], p2, "must")[0].met


def test_synonyms_map_llm_to_agents():
    ev = aw.match_evidence(["LLMs"], PROFILE, "must")[0]
    assert ev.met and any("LangGraph" in l or "LLM" in l for l in ev.lines)


def test_fit_score_and_decision():
    ev = aw.match_evidence(["Python", "RAG"], PROFILE, "must") + aw.match_evidence(["Kubernetes"], PROFILE, "nice")
    assert aw.fit_score(ev) == 60
    assert aw.decide(aw.GateResult("ELIGIBLE", []), 60) == "APPLY"
    assert aw.decide(aw.GateResult("CHECK", []), 80) == "APPLY AFTER CHECKING"
    assert aw.decide(aw.GateResult("NOT_ELIGIBLE", []), 100) == "SKIP"
    assert aw.decide(aw.GateResult("ELIGIBLE", []), 10) == "SKIP"


# ---------------- claim guard ----------------
def test_claim_guard_flags_invented_skills_and_numbers():
    note = "I built RAG pipelines with LangChain and Python, improving accuracy by 95%."
    flagged = aw.unsupported_claims(note, PROFILE)
    assert "langchain" in flagged and "rag" in flagged and "95%" in flagged
    assert "python" not in flagged


def test_claim_guard_accepts_grounded_note():
    note = "I built 2 LLM agent workflows with LangGraph and MCP, cutting review time by 40%, and ship FastAPI services."
    assert aw.unsupported_claims(note, PROFILE) == []


# ---------------- client ----------------
def test_client_retries_on_503(monkeypatch):
    monkeypatch.setattr(aw.time, "sleep", lambda s: None)
    t = fake_transport([(503, "", None), (200, "hello", "stop")])
    c = aw.BharatCodeClient(api_key="k", transport=t)
    assert c.chat("s", "u") == "hello"
    assert len(t.calls) == 2
    url, payload = t.calls[0]
    assert url.endswith("/chat/completions") and payload["model"] == "qwen-3.8-27b"


def test_client_rejects_truncated_output():
    c = aw.BharatCodeClient(api_key="k", transport=fake_transport([(200, "partial", "length")]))
    with pytest.raises(aw.BharatCodeError, match="cut short"):
        c.chat("s", "u")


def test_client_surfaces_http_errors():
    c = aw.BharatCodeClient(api_key="k", transport=fake_transport([(401, "", None)]))
    with pytest.raises(aw.BharatCodeError, match="401"):
        c.chat("s", "u")


def test_missing_key_message(monkeypatch):
    monkeypatch.delenv("BHARATCODE_API_KEY", raising=False)
    with pytest.raises(aw.BharatCodeError, match="BHARATCODE_API_KEY"):
        aw.BharatCodeClient().chat("s", "u")


def test_parse_json_handles_code_fences():
    assert aw._parse_json('```json\n{"a": 1}\n```') == {"a": 1}


# ---------------- end to end ----------------
def test_end_to_end_skips_ineligible_listing_without_writing_note():
    t = fake_transport([(200, EXTRACTED, "stop")])
    r = aw.analyse(LISTING, PROFILE, client=aw.BharatCodeClient(api_key="k", transport=t), today=NOV_2026)
    assert r.decision == "SKIP" and r.cover_note == ""
    assert len(t.calls) == 1  # no wasted tokens on a note for an ineligible role
    md = aw.render_markdown(r)
    assert "MISSING" in md and "RAG" in md


def test_end_to_end_eligible_listing_writes_guarded_note():
    listing = "Remote AI intern, open to the 2028 batch. Python, REST APIs, LLMs, Git, RAG."
    extracted = json.loads(EXTRACTED) | {"eligibility": ["open to the 2028 batch"], "work_mode": "remote"}
    bad_note = "I used LangChain for RAG and Python."
    good_note = "I built 2 LLM agent workflows with LangGraph and MCP, cutting review time by 40%."
    t = fake_transport([(200, json.dumps(extracted), "stop"), (200, bad_note, "stop"), (200, good_note, "stop")])
    r = aw.analyse(listing, PROFILE, client=aw.BharatCodeClient(api_key="k", transport=t), today=NOV_2026)
    assert r.decision == "APPLY"
    assert r.cover_note == good_note and r.unsupported_claims == []
    assert "NOT in the evidence" in t.calls[2][1]["messages"][1]["content"]  # repair pass happened


def test_cli_json_output(tmp_path, monkeypatch, capsys):
    t = fake_transport([(200, EXTRACTED, "stop")])
    real = aw.BharatCodeClient

    class Fake(real):
        def __init__(self):
            super().__init__(api_key="k", transport=t)

    monkeypatch.setattr(aw, "BharatCodeClient", Fake)
    code = aw.main([str(HERE / "sample_listing.txt"), "--profile", str(HERE / "sample_profile.md"), "--json"])
    assert code == 0
    out = json.loads(capsys.readouterr().out)
    assert out["decision"] == "SKIP" and out["gate"]["verdict"] == "NOT_ELIGIBLE"
