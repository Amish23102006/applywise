# ApplyWise: an evidence-grounded internship application copilot

> Built for the **BharatCode Build League**, on the BharatCode API (`qwen-3.8-27b`, OpenAI-compatible).
> Direction: **Work & automation**

Students in India apply to dozens of internships every week across Wellfound, Internshala, Unstop, Instahyre and company portals. Most of that time goes on three tasks, each of which goes wrong in a familiar way:

1. **Reading the fine print.** Which graduation batch is allowed? Final-year only? Onsite in another city? Immediate start? A missed line makes the whole application wasted.
2. **Matching the requirements.** Which of my projects actually proves "REST APIs" or "LLMs"?
3. **Writing the note.** Generic AI cover letters invent skills you don't have, and those claims fall apart in the interview.

ApplyWise is a harness that combines a BharatCode model with deterministic checks, so the LLM does the language work and plain code handles the facts.

```
listing + your CV
      |
      v
1. Eligibility gate (deterministic)   cohort ("2026/27 grads only"), final/pre-final year, IIT-only,
                                      onsite vs remote, immediate start, unpaid, fee requests (scam flag)
2. Requirement extraction (BharatCode) must-have / nice-to-have skills, location, duration, stipend -> strict JSON
3. Evidence matching (deterministic)   every requirement -> the exact lines of YOUR CV that prove it, or MISSING
4. Fit score + decision                APPLY / APPLY AFTER CHECKING / STRETCH / SKIP
5. Evidence-only note (BharatCode)     the model sees only the matched evidence lines
6. Claim guard (deterministic)         flags any tech term or number in the note that is not in your CV,
                                      triggers one repair pass, then warns about anything still unsupported
```

If the gate says **SKIP**, no note is generated, so no tokens are spent on a role you can't take.

## Quick start

```bash
git clone https://github.com/Amish23102006/applywise && cd applywise
export BHARATCODE_API_KEY=bc_live_...      # from https://bharatcode.ai/account (keep it server-side)

# CLI: standard library only, nothing to install
python applywise.py sample_listing.txt --profile sample_profile.md \
    --location Indore --open-to remote --available-from "April 2027"

# Web UI
pip install -r requirements.txt
uvicorn web:app --reload                   # open http://127.0.0.1:8000
```

`--json` prints machine-readable output. `--no-note` skips the cover note.

### Example output (sample files, fictional data)

```
# AI Engineer Intern - Acme Robotics
Decision: SKIP | Fit score: 77/100 | Eligibility: NOT_ELIGIBLE

## Eligibility check
- Listing targets graduation year(s) [2026, 2027]; you graduate in 2028.
- Role is onsite; confirm you can relocate or commute.
- Role wants an immediate start; you are available from April 2027.

## Requirement -> your evidence
| Python     | must | Languages: Python, JavaScript, C++, SQL |
| LLMs       | must | AI Intern, Example Corp: built 2 LLM agent workflows with LangGraph and MCP ... |
| RAG        | must | MISSING - build or learn this before claiming it |
| TypeScript | nice | MISSING - build or learn this before claiming it |

## Gaps to close
- RAG
```

The tests reproduce this output exactly. With a live key, the score can shift slightly depending on which skills the model extracts.

## How it uses BharatCode

| Call | Endpoint | Settings |
|---|---|---|
| Requirement extraction | `POST /chat/completions` | `response_format: json_object`, `temperature 0.2`, `reasoning_effort: none` |
| Evidence-only note (+ at most one repair pass) | `POST /chat/completions` | `max_tokens 600` |

Client behaviour follows the BharatCode API guidance:
- The key is read only from `BHARATCODE_API_KEY`. It is never logged, and in the web UI it never reaches the browser.
- `503` (model busy) is retried with exponential backoff.
- `finish_reason: "length"` is treated as truncated output and raises an error instead of being silently trusted.
- Requests are stateless, and no user data is stored.

## Design choices

- **Deterministic first.** Cohort years, year of study and fee red flags are checked with regex rules that are unit-tested, not left to the model to guess.
- **Grounding by construction.** The note prompt contains only CV lines that matched a requirement, and the claim guard verifies the output against the full CV.
- **Honest gaps.** Missing requirements are listed as gaps to close, never smoothed over.
- **Zero-dependency CLI.** Python standard library only. FastAPI is needed only for the web UI.

## Tests

```bash
pip install pytest fastapi httpx python-multipart
pytest -q        # 30 tests, fully offline: the BharatCode API is replaced by a fake transport
```

The tests cover cohort and year-of-study parsing, onsite and immediate-start checks, scam flags, whole-word skill matching, synonym mapping, scoring, the claim guard, 503 retry, truncated-output handling, the end-to-end flow (including the repair pass), CLI JSON output and HTML escaping in the web UI.

## Limitations and next steps

- Eligibility rules are English-language regex heuristics, so always read the listing too.
- Evidence matching is lexical with a synonym table. Embedding-based matching through BharatCode is a natural next step.
- Planned: a browser extension that runs ApplyWise on the listing you are viewing, plus batch mode over a CSV of listings.

## License

MIT
