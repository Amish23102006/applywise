"""ApplyWise - evidence-grounded internship application copilot powered by BharatCode.

Pipeline
  1. Deterministic eligibility gate  (graduation cohort, year of study, location, duration)
  2. LLM requirement extraction        (BharatCode qwen-3.8-27b, structured JSON)
  3. Deterministic evidence matching   (every requirement -> lines from YOUR profile, or "missing")
  4. LLM cover note                    (may only use the evidence lines it is given)
  5. Claim guard                       (flags any tech term in the note that is not in your profile)

Only the Python standard library is needed for the CLI.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date

BASE_URL = os.environ.get("BHARATCODE_BASE_URL", "https://bharatcode.ai/api/model/v1")
MODEL = os.environ.get("BHARATCODE_MODEL", "qwen-3.8-27b")
MAX_INPUT_CHARS = 20_000


# --------------------------------------------------------------------------- #
# BharatCode client (OpenAI-compatible /chat/completions)
# --------------------------------------------------------------------------- #
class BharatCodeError(RuntimeError):
    pass


class BharatCodeClient:
    """Minimal client for the BharatCode OpenAI-compatible API.

    - Reads the key from BHARATCODE_API_KEY (never hard-code or log it).
    - Retries 503 (model busy) with backoff.
    - Treats finish_reason == "length" as truncated output instead of trusting it.
    """

    def __init__(self, api_key: str | None = None, base_url: str = BASE_URL, model: str = MODEL,
                 timeout: float = 120.0, retries: int = 3, transport=None):
        self.api_key = api_key or os.environ.get("BHARATCODE_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self._transport = transport or self._http_post

    def _http_post(self, url: str, payload: dict) -> tuple[int, dict]:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode(),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                return e.code, json.loads(body)
            except json.JSONDecodeError:
                return e.code, {"error": body[:500]}

    def chat(self, system: str, user: str, max_tokens: int = 2000, json_mode: bool = False) -> str:
        if not self.api_key and self._transport == self._http_post:
            raise BharatCodeError("Set BHARATCODE_API_KEY (get it from https://bharatcode.ai/account).")
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_tokens": max_tokens,
            "temperature": 0.2,
            "reasoning_effort": "none",
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        delay = 2.0
        for attempt in range(self.retries + 1):
            status, data = self._transport(f"{self.base_url}/chat/completions", payload)
            if status == 503 and attempt < self.retries:
                time.sleep(delay)
                delay *= 2
                continue
            if status != 200:
                raise BharatCodeError(f"BharatCode API error {status}: {str(data)[:300]}")
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
            if choice.get("finish_reason") == "length":
                raise BharatCodeError("Output was cut short (finish_reason=length); raise max_tokens.")
            return content
        raise BharatCodeError("BharatCode model stayed busy (503) after retries.")


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class Profile:
    text: str
    grad_year: int | None = None
    location: str | None = None
    open_to: list[str] = field(default_factory=list)  # e.g. ["remote", "bengaluru"]
    available_from: str | None = None

    @property
    def lines(self) -> list[str]:
        out = []
        for raw in self.text.splitlines():
            line = raw.strip(" \t-*•#|")
            if len(line) >= 3:
                out.append(line)
        return out


@dataclass
class GateResult:
    verdict: str  # ELIGIBLE | CHECK | NOT_ELIGIBLE
    reasons: list[str]


@dataclass
class Evidence:
    requirement: str
    kind: str  # must | nice
    lines: list[str]

    @property
    def met(self) -> bool:
        return bool(self.lines)


@dataclass
class Report:
    role: str
    company: str
    gate: GateResult
    evidence: list[Evidence]
    fit_score: int
    decision: str
    cover_note: str
    unsupported_claims: list[str]
    facts: dict

    def to_dict(self) -> dict:
        d = asdict(self)
        d["evidence"] = [asdict(e) | {"met": e.met} for e in self.evidence]
        return d


# --------------------------------------------------------------------------- #
# Profile parsing
# --------------------------------------------------------------------------- #
def load_profile(text: str, grad_year: int | None = None, location: str | None = None,
                 open_to: list[str] | None = None, available_from: str | None = None) -> Profile:
    text = text[:MAX_INPUT_CHARS]
    if grad_year is None:
        years = [int(y) for y in re.findall(r"20[2-3]\d", " ".join(
            l for l in text.splitlines() if re.search(r"expected|graduat|b\.?tech|bachelor|class of", l, re.I)))]
        future = [y for y in years if y >= 2024]
        grad_year = max(future) if future else None
    return Profile(text=text, grad_year=grad_year, location=location,
                   open_to=[o.lower() for o in (open_to or [])], available_from=available_from)


# --------------------------------------------------------------------------- #
# 1. Deterministic eligibility gate
# --------------------------------------------------------------------------- #
_COHORT_PATTERNS = [
    r"(?:graduates?|batch|pass[- ]?outs?|class)\s*(?:of|from|in)?\s*(?:the\s*)?((?:20\d\d)(?:\s*(?:[/&,-]|and|or)\s*(?:20)?\d\d)*)",
    r"((?:20\d\d)(?:\s*(?:[/&,-]|and|or)\s*(?:20)?\d\d)*)\s*(?:batch|graduates?|pass[- ]?outs?|grads?)",
]


def _expand_years(group: str) -> set[int]:
    years = set()
    for tok in re.split(r"[/&,\-\s]+|\band\b|\bor\b", group):
        if re.fullmatch(r"20\d\d", tok):
            years.add(int(tok))
        elif re.fullmatch(r"\d\d", tok):
            years.add(2000 + int(tok))
    return years


def _study_year(grad_year: int, today: date) -> int:
    """Year of study (1-4+) for a 4-year degree, assuming the academic year starts in July."""
    academic_start = today.year if today.month >= 7 else today.year - 1
    return 4 - (grad_year - 1 - academic_start)


def eligibility_gate(listing: str, profile: Profile, today: date | None = None) -> GateResult:
    today = today or date.today()
    reasons: list[str] = []
    verdict = "ELIGIBLE"
    low = listing.lower()

    def downgrade(level: str, reason: str):
        nonlocal verdict
        order = {"ELIGIBLE": 0, "CHECK": 1, "NOT_ELIGIBLE": 2}
        if order[level] > order[verdict]:
            verdict = level
        reasons.append(reason)

    # Graduation cohort restrictions, e.g. "2026 batch", "graduates from the 2026/27"
    cohort: set[int] = set()
    for pat in _COHORT_PATTERNS:
        for m in re.finditer(pat, listing, re.I):
            cohort |= _expand_years(m.group(1))
    if cohort and profile.grad_year:
        if profile.grad_year not in cohort:
            downgrade("NOT_ELIGIBLE",
                      f"Listing targets graduation year(s) {sorted(cohort)}; you graduate in {profile.grad_year}.")
        else:
            reasons.append(f"Your {profile.grad_year} graduation matches the stated cohort.")

    # Year-of-study restrictions
    if profile.grad_year:
        year = _study_year(profile.grad_year, today)
        if re.search(r"final[- ]year|8th semester|fourth[- ]year|4th[- ]year", low) and year < 4 \
                and not re.search(r"(any|all) years?|1st|first[- ]year", low):
            downgrade("NOT_ELIGIBLE" if re.search(r"only|eligib", low) else "CHECK",
                      f"Listing asks for final-year students; you are in year {year}.")
        elif re.search(r"pre[- ]final", low) and year < 3:
            downgrade("CHECK", f"Listing asks for pre-final-year students; you are in year {year}.")

    # Institution restrictions
    if re.search(r"\b(iit|nit|iiit|bits)\b[^.]{0,40}(only|or similar|or equivalent|tier[- ]?1)", low):
        downgrade("CHECK", "Listing prefers specific institutions (IIT/NIT/BITS or similar).")

    # Location / mode
    onsite = re.search(r"on[- ]?site|in[- ]office|work from office|wfo|5[- ]day", low)
    remote = re.search(r"remote|work from home|wfh", low)
    if onsite and not remote:
        loc_ok = profile.location and profile.location.lower() in low
        open_ok = any(o in low for o in profile.open_to if o != "remote")
        if not (loc_ok or open_ok):
            downgrade("CHECK", "Role is onsite; confirm you can relocate or commute.")

    # Duration / immediate start
    if re.search(r"immediate(ly)?\s*(join|start)|join(ing)? immediately|immediate joiners?", low) and profile.available_from:
        downgrade("CHECK", f"Role wants an immediate start; you are available from {profile.available_from}.")
    dur = re.search(r"(\d+)\s*(?:-|to)?\s*(\d+)?\s*months?", low)
    if dur:
        reasons.append(f"Duration mentioned: {dur.group(0)}.")

    # Unpaid / fee red flags
    if re.search(r"unpaid|no stipend|stipend:\s*(?:nil|0)\b", low):
        downgrade("CHECK", "Listing is unpaid.")
    if re.search(r"registration fee|training fee|pay (?:a|the) fee|security deposit", low):
        downgrade("NOT_ELIGIBLE", "Listing asks candidates to pay a fee - likely a scam.")

    if not reasons:
        reasons.append("No conflicting cohort, location or duration rule found.")
    return GateResult(verdict, reasons)


# --------------------------------------------------------------------------- #
# 2. Requirement extraction (LLM, structured JSON)
# --------------------------------------------------------------------------- #
EXTRACT_SYSTEM = """You extract hiring requirements from a job or internship listing.
Return ONLY a JSON object with these keys:
  "company": string, "role": string,
  "must_have": list of short skill/requirement strings (max 12),
  "nice_to_have": list of short skill/requirement strings (max 10),
  "location": string, "work_mode": "remote" | "onsite" | "hybrid" | "unknown",
  "duration": string, "stipend": string, "start": string,
  "eligibility": list of strings quoting any cohort, year-of-study or institution rules.
Use "" or [] when the listing does not say. Never invent facts that are not in the listing.
Keep each skill to 1-4 words (e.g. "Python", "REST APIs", "React", "LLM agents")."""


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise BharatCodeError("Model did not return JSON.")
    return json.loads(m.group(0))


def extract_requirements(client: BharatCodeClient, listing: str) -> dict:
    raw = client.chat(EXTRACT_SYSTEM, listing[:MAX_INPUT_CHARS], max_tokens=1500, json_mode=True)
    data = _parse_json(raw)
    for k in ("must_have", "nice_to_have", "eligibility"):
        v = data.get(k) or []
        data[k] = [str(x).strip() for x in v if str(x).strip()][:12]
    for k in ("company", "role", "location", "work_mode", "duration", "stipend", "start"):
        data[k] = str(data.get(k) or "").strip()
    return data


# --------------------------------------------------------------------------- #
# 3. Deterministic evidence matching
# --------------------------------------------------------------------------- #
SYNONYMS = {
    "llm": ["llm", "large language model", "gpt", "langgraph", "agentic", "mcp"],
    "llms": ["llm", "large language model", "langgraph", "agentic", "mcp"],
    "rest apis": ["rest api", "fastapi", "express", "api"],
    "apis": ["api", "fastapi", "express"],
    "machine learning": ["machine learning", "ml", "scikit-learn", "pytorch"],
    "deep learning": ["pytorch", "deep learning", "resnet", "autoencoder"],
    "ci/cd": ["ci/cd", "github actions", "ci tests", "ci "],
    "testing": ["pytest", "node:test", "tests", "k6"],
    "cloud": ["aws", "azure", "google cloud", "gcp", "render", "vercel", "kubernetes"],
    "databases": ["sql", "sqlite", "mysql", "postgres", "mongodb"],
    "sql": ["sql", "sqlite", "mysql", "postgres"],
    "computer vision": ["computer vision", "opencv", "image", "mvtec"],
    "nlp": ["nlp", "text classification", "imdb", "ag news", "language model"],
    "data structures": ["data structures", "algorithms", "codeforces"],
    "algorithms": ["algorithms", "data structures", "codeforces"],
    "linux": ["linux", "bash", "shell"],
    "javascript": ["javascript", "node.js", "react", "express"],
    "node.js": ["node.js", "nodejs", "express"],
    "react": ["react"],
    "docker": ["docker"],
    "git": ["git", "github"],
}

_STOP = {"and", "or", "with", "of", "the", "a", "an", "in", "to", "for", "experience", "knowledge",
         "understanding", "basic", "strong", "good", "familiarity", "proficiency", "skills", "skill"}


def _terms(requirement: str) -> list[str]:
    req = requirement.lower().strip()
    if req in SYNONYMS:
        return SYNONYMS[req]
    words = [w for w in re.findall(r"[a-z0-9+#./-]+", req) if w not in _STOP and len(w) > 1]
    terms = [req] + words
    for w in words:
        terms += SYNONYMS.get(w, [])
    return list(dict.fromkeys(terms))


def _contains(line: str, term: str) -> bool:
    if len(term) <= 3 or re.fullmatch(r"[a-z]+", term):
        return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", line) is not None
    return term in line


def match_evidence(requirements: list[str], profile: Profile, kind: str, max_lines: int = 2) -> list[Evidence]:
    out = []
    lowered = [(l, l.lower()) for l in profile.lines]
    for req in requirements:
        terms = _terms(req)
        hits = []
        for orig, low in lowered:
            if any(_contains(low, t) for t in terms):
                hits.append(orig)
        # prefer longer, more specific lines (project/experience bullets)
        hits = sorted(dict.fromkeys(hits), key=len, reverse=True)[:max_lines]
        out.append(Evidence(req, kind, hits))
    return out


def fit_score(evidence: list[Evidence]) -> int:
    must = [e for e in evidence if e.kind == "must"]
    nice = [e for e in evidence if e.kind == "nice"]
    if not must and not nice:
        return 0
    m = sum(e.met for e in must) / len(must) if must else None
    n = sum(e.met for e in nice) / len(nice) if nice else None
    if m is None:
        return round(100 * n)
    if n is None:
        return round(100 * m)
    return round(100 * (0.8 * m + 0.2 * n))


def decide(gate: GateResult, score: int) -> str:
    if gate.verdict == "NOT_ELIGIBLE":
        return "SKIP"
    if score >= 60 and gate.verdict == "ELIGIBLE":
        return "APPLY"
    if score >= 60:
        return "APPLY AFTER CHECKING"
    if score >= 40:
        return "STRETCH - apply only if you can close the gaps"
    return "SKIP"


# --------------------------------------------------------------------------- #
# 4. Evidence-only cover note (LLM) + 5. claim guard
# --------------------------------------------------------------------------- #
NOTE_SYSTEM = """You write short, specific internship application notes.
Rules:
- Use ONLY facts from the EVIDENCE lines. Do not add any skill, tool, number, employer or project that is not in them.
- 70-110 words, first person, plain professional tone, no greeting line like "Dear Sir/Madam", no emojis.
- Open with one line on why this company/role fits, then 2-3 concrete proof points with numbers from the evidence.
- End with one short closing sentence. Output the note text only."""

TECH_TERMS = [
    "python", "java", "javascript", "typescript", "c++", "go", "rust", "react", "next.js", "node.js", "express",
    "fastapi", "django", "flask", "streamlit", "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy",
    "opencv", "langchain", "langgraph", "llamaindex", "mcp", "rag", "docker", "kubernetes", "aws", "azure",
    "gcp", "google cloud", "sql", "mysql", "postgresql", "mongodb", "redis", "kafka", "spark", "airflow",
    "github actions", "terraform", "prometheus", "grafana", "tailwind", "prisma", "graphql", "hugging face",
    "transformers", "bert", "vector database", "pinecone", "faiss", "selenium", "postman", "linux", "bash",
]


def unsupported_claims(note: str, profile: Profile) -> list[str]:
    note_l, prof_l = note.lower(), profile.text.lower()
    bad = []
    for t in TECH_TERMS:
        if _contains(note_l, t) and not _contains(prof_l, t):
            bad.append(t)
    for num in set(re.findall(r"\b\d+(?:\.\d+)?%?", note)):
        if num not in profile.text and len(num) > 1:
            bad.append(num)
    return sorted(set(bad))


def write_cover_note(client: BharatCodeClient, req: dict, evidence: list[Evidence], profile: Profile) -> tuple[str, list[str]]:
    lines = []
    for e in evidence:
        lines += e.lines
    lines = list(dict.fromkeys(lines))[:14]
    if not lines:
        return "", []
    user = (f"COMPANY: {req.get('company') or 'the company'}\nROLE: {req.get('role') or 'the internship'}\n"
            f"WHAT THEY WANT: {', '.join(req.get('must_have', [])[:8])}\n\nEVIDENCE:\n- " + "\n- ".join(lines))
    note = client.chat(NOTE_SYSTEM, user, max_tokens=600).strip()
    flagged = unsupported_claims(note, profile)
    if flagged:  # one repair pass, then report whatever is still unsupported
        fix = (user + f"\n\nYOUR DRAFT:\n{note}\n\nThese items are NOT in the evidence: {', '.join(flagged)}. "
               "Rewrite the note without them, following all rules.")
        note = client.chat(NOTE_SYSTEM, fix, max_tokens=600).strip()
        flagged = unsupported_claims(note, profile)
    return note, flagged


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def analyse(listing: str, profile: Profile, client: BharatCodeClient | None = None, today: date | None = None,
            write_note: bool = True) -> Report:
    client = client or BharatCodeClient()
    listing = listing[:MAX_INPUT_CHARS]
    gate = eligibility_gate(listing, profile, today=today)
    req = extract_requirements(client, listing)
    for rule in req.get("eligibility", []):
        sub = eligibility_gate(rule, profile, today=today)
        for r in sub.reasons:
            if r not in gate.reasons and sub.verdict != "ELIGIBLE":
                gate.reasons.append(r)
        order = ["ELIGIBLE", "CHECK", "NOT_ELIGIBLE"]
        if order.index(sub.verdict) > order.index(gate.verdict):
            gate.verdict = sub.verdict
    evidence = match_evidence(req["must_have"], profile, "must") + match_evidence(req["nice_to_have"], profile, "nice")
    score = fit_score(evidence)
    decision = decide(gate, score)
    note, flagged = ("", [])
    if write_note and decision != "SKIP":
        note, flagged = write_cover_note(client, req, evidence, profile)
    facts = {k: req.get(k, "") for k in ("location", "work_mode", "duration", "stipend", "start")}
    return Report(req.get("role", ""), req.get("company", ""), gate, evidence, score, decision, note, flagged, facts)


def render_markdown(r: Report) -> str:
    out = [f"# {r.role or 'Role'} - {r.company or 'Company'}", "",
           f"**Decision:** {r.decision}  |  **Fit score:** {r.fit_score}/100  |  **Eligibility:** {r.gate.verdict}", ""]
    facts = " | ".join(f"{k}: {v}" for k, v in r.facts.items() if v)
    if facts:
        out += [f"_{facts}_", ""]
    out += ["## Eligibility check"] + [f"- {x}" for x in r.gate.reasons] + [""]
    out += ["## Requirement -> your evidence", "", "| Requirement | Type | Evidence from your profile |", "|---|---|---|"]
    for e in r.evidence:
        ev = "<br>".join(e.lines) if e.lines else "**MISSING** - build or learn this before claiming it"
        out.append(f"| {e.requirement} | {e.kind} | {ev} |")
    gaps = [e.requirement for e in r.evidence if not e.met and e.kind == "must"]
    if gaps:
        out += ["", "## Gaps to close", ""] + [f"- {g}" for g in gaps]
    if r.cover_note:
        out += ["", "## Evidence-only application note", "", r.cover_note]
        if r.unsupported_claims:
            out += ["", f"> Warning: remove these unsupported items before sending: {', '.join(r.unsupported_claims)}"]
    return "\n".join(out) + "\n"


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="ApplyWise: evidence-grounded internship copilot (BharatCode).")
    p.add_argument("listing", help="path to a text file containing the job/internship listing")
    p.add_argument("--profile", required=True, help="path to your CV/profile as text or markdown")
    p.add_argument("--grad-year", type=int, help="your graduation year (auto-detected if omitted)")
    p.add_argument("--location", help="your current city, e.g. Indore")
    p.add_argument("--open-to", default="remote", help="comma-separated places you can work, e.g. remote,bengaluru")
    p.add_argument("--available-from", help="e.g. 'April 2027'")
    p.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    p.add_argument("--no-note", action="store_true", help="skip the cover note")
    a = p.parse_args(argv)
    profile = load_profile(_read(a.profile), a.grad_year, a.location,
                           [x.strip() for x in a.open_to.split(",") if x.strip()], a.available_from)
    try:
        report = analyse(_read(a.listing), profile, write_note=not a.no_note)
    except BharatCodeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(report.to_dict(), indent=2) if a.json else render_markdown(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
