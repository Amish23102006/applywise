"""ApplyWise web UI - run with:  uvicorn web:app --reload   (needs: pip install fastapi uvicorn)

The BharatCode key stays on the server (BHARATCODE_API_KEY env var); the browser never sees it.
Nothing is stored: the profile and listing live only for the duration of one request.
"""
from __future__ import annotations

import html

from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse

from applywise import MAX_INPUT_CHARS, BharatCodeError, analyse, load_profile

app = FastAPI(title="ApplyWise", docs_url=None, redoc_url=None)

PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ApplyWise</title>
<style>
:root{{--bg:#f7f7f5;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--acc:#1f6f5c;--warn:#a5481b}}
@media (prefers-color-scheme:dark){{:root{{--bg:#141413;--fg:#ecebe6;--mut:#a3a29b;--card:#1d1d1b;--line:#33322e;--acc:#5fc3a6;--warn:#f0956a}}}}
body{{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}}
main{{max-width:960px;margin:0 auto;padding:24px 16px}}h1{{margin:0 0 4px}}p.sub{{color:var(--mut);margin:0 0 20px}}
form{{display:grid;gap:12px}}.row{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}
@media(max-width:700px){{.row{{grid-template-columns:1fr}}}}
textarea,input{{width:100%;box-sizing:border-box;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:10px;font:inherit}}
textarea{{min-height:220px}}button{{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:12px 18px;font-weight:600;cursor:pointer}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px;margin-top:20px}}
table{{width:100%;border-collapse:collapse}}td,th{{border-top:1px solid var(--line);padding:8px;text-align:left;vertical-align:top}}
.miss{{color:var(--warn);font-weight:600}}.pill{{display:inline-block;padding:2px 10px;border-radius:99px;border:1px solid var(--line);margin-right:6px}}
.note{{white-space:pre-wrap}}</style></head><body><main>
<h1>ApplyWise</h1><p class="sub">Paste an internship listing and your CV. ApplyWise checks eligibility, maps every requirement to evidence
from your own CV, and drafts a note that only uses that evidence. Powered by BharatCode.</p>
<form method="post" action="/analyse">
<div class="row"><label>Job / internship listing<textarea name="listing" required maxlength="{max}">{listing}</textarea></label>
<label>Your CV / profile (plain text)<textarea name="profile" required maxlength="{max}">{profile}</textarea></label></div>
<div class="row"><label>Graduation year<input name="grad_year" inputmode="numeric" value="{grad_year}"></label>
<label>Available from<input name="available_from" placeholder="April 2027" value="{available_from}"></label></div>
<div class="row"><label>Current city<input name="location" placeholder="Indore" value="{location}"></label>
<label>Open to (comma separated)<input name="open_to" value="{open_to}"></label></div>
<button type="submit">Analyse</button></form>{result}</main></body></html>"""


def _page(result: str = "", **vals) -> str:
    defaults = {"listing": "", "profile": "", "grad_year": "", "available_from": "", "location": "", "open_to": "remote"}
    defaults.update({k: html.escape(str(v)) for k, v in vals.items() if v is not None})
    return PAGE.format(result=result, max=MAX_INPUT_CHARS, **defaults)


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return _page()


@app.post("/analyse", response_class=HTMLResponse)
def analyse_view(listing: str = Form(...), profile: str = Form(...), grad_year: str = Form(""),
                 available_from: str = Form(""), location: str = Form(""), open_to: str = Form("remote")) -> str:
    gy = int(grad_year) if grad_year.strip().isdigit() else None
    prof = load_profile(profile, gy, location or None, [x.strip() for x in open_to.split(",") if x.strip()],
                        available_from or None)
    vals = dict(listing=listing, profile=profile, grad_year=grad_year, available_from=available_from,
                location=location, open_to=open_to)
    try:
        r = analyse(listing, prof)
    except BharatCodeError as e:
        return _page(f'<div class="card miss">{html.escape(str(e))}</div>', **vals)
    esc = html.escape
    rows = "".join(
        f"<tr><td>{esc(e.requirement)}</td><td>{e.kind}</td><td>"
        + ("<br>".join(esc(l) for l in e.lines) if e.lines else '<span class="miss">Missing</span>')
        + "</td></tr>" for e in r.evidence)
    reasons = "".join(f"<li>{esc(x)}</li>" for x in r.gate.reasons)
    note = ""
    if r.cover_note:
        warn = (f'<p class="miss">Remove before sending: {esc(", ".join(r.unsupported_claims))}</p>'
                if r.unsupported_claims else "")
        note = f'<div class="card"><h3>Evidence-only note</h3><p class="note">{esc(r.cover_note)}</p>{warn}</div>'
    card = (f'<div class="card"><h2>{esc(r.role or "Role")} - {esc(r.company or "Company")}</h2>'
            f'<span class="pill">Decision: <b>{esc(r.decision)}</b></span><span class="pill">Fit {r.fit_score}/100</span>'
            f'<span class="pill">Eligibility: {esc(r.gate.verdict)}</span><ul>{reasons}</ul></div>'
            f'<div class="card"><h3>Requirement to evidence</h3><table><tr><th>Requirement</th><th>Type</th>'
            f'<th>Your evidence</th></tr>{rows}</table></div>{note}')
    return _page(card, **vals)
