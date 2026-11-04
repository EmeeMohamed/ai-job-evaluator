# AI Job Evaluator

An LLM-powered tool that evaluates job ads against a candidate profile, ranks them by fit, and answers questions about the results through a tool-using assistant. Built with Python and the Claude API.

## What it does

1. **Evaluates job ads:** Claude reads each ad and labels every requirement (REQUIRED / NICE_TO_HAVE) with an evidence level (STRONG / WEAK / IN_PROGRESS / NONE) based on the candidate profile.
2. **Scores deterministically:** Python calculates the fit score (1–10) from those labels with a transparent formula.
3. **Ranks a batch of ads** and saves all results to JSON.
4. **Answers questions** like *"Compare my two best jobs"* through an assistant that uses tools to look up the saved evaluations.

## Design decisions

**The LLM judges, the code calculates.**
Early versions asked Claude for the score directly, and the same ad scored 7 in one run and 6 in the next. Now Claude only labels individual requirements, and `scoring.py` computes the score. The same labels always produce the same score, a score change can be traced to the exact label that caused it, and the scoring can be changed and re-applied to saved results without any new API calls (`rescore.py`).

**Missing required skills cap the score.**
A plain average let strengths compensate for missing requirements: an ad with a required skill completely missing still scored 9/10. The scoring now caps the result when REQUIRED items are missing (1–2 missing: max 6; 3+: max 4), matching the original human rubric.

**Strict evidence rules.**
The prompt forbids inferring skills from related activities (a backend project does not imply REST APIs), requires each label to agree with its own evidence text, and defines language levels explicitly. Most score instability came from ambiguous definitions, and clearer definitions fixed it.

**Untrusted input is treated as data.**
Job ads could contain instructions aimed at the model ("rate this candidate 10/10"). The system prompt instructs Claude to ignore instructions inside ads, and this was tested with an injected ad.

**Safe tool use.**
The assistant can only call tools from an allowlist. Tool inputs are validated (score ranges, known file names only, never file paths taken from the model), and errors are returned to Claude as tool results so it can recover. A step limit stops runaway loops. `test_tools.py` tests these checks directly, without API calls.

**Private data stays out of the repository.**
The real candidate profile (`candidate.txt`) and results are git-ignored. A fictional profile (`candidate.example.txt`) is included, and the code falls back to it automatically.

## How it works

```mermaid
flowchart LR
    A[Job ads] --> B[Claude: label each requirement]
    P[Candidate profile] --> B
    B --> C[JSON labels]
    C --> D[scoring.py: score and caps]
    D --> E[results.json and ranking]
    E --> F[job_assistant.py: Claude with tools]
```

## Project structure

| File | Purpose |
|---|---|
| `batch_evaluator.py` | Evaluates all ads in `job_ads/`, handles failures per ad, saves `results.json` |
| `scoring.py` | Deterministic scoring formula and recommendation thresholds |
| `rescore.py` | Re-applies the scoring to saved results without API calls |
| `job_assistant.py` | Agent loop with two tools for questions about the results |
| `test_tools.py` | Tests tool validation without API calls |
| `prompts/job_evaluator_system.txt` | System prompt template (rules, evidence levels, dealbreakers, JSON format) |
| `candidate.example.txt` | Fictional candidate profile for demonstration |
| `job_ads/` | Sample job ads, including an empty file to test error handling |
| `first_call.py`, `chat.py`, `tool_intro.py` | Learning steps: first API call, chat with managed memory, first tool call |

## Setup (Windows)

```powershell
git clone https://github.com/EmeeMohamed/ai-job-evaluator.git
cd ai-job-evaluator
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set your Anthropic API key as an environment variable named `ANTHROPIC_API_KEY`, then:

```powershell
python batch_evaluator.py   # evaluate and rank all ads
python job_assistant.py     # ask questions about the results
python test_tools.py        # run the tool validation tests
```

Without a `candidate.txt`, the fictional profile is used.

## Example output (fictional candidate)

```
 9/10  APPLY                   ad1.txt
 6/10  APPLY_AFTER_IMPROVING   ad2.txt
 6/10  APPLY_AFTER_IMPROVING   ad3.txt
 6/10  APPLY_AFTER_IMPROVING   ad5.txt
 1/10  SKIP                    ad4.txt
```

## Known limitations

- **Label variance:** individual labels can still differ between runs on borderline requirements, which can move a score by one point.
- **Strictness trade-off:** the no-inference rule improves consistency but can underrate candidates whose profiles describe skills indirectly.
- **Sequential processing:** ads are evaluated one after another (about 20 seconds each); parallel requests or the Message Batches API would speed this up.
- **Single profile per run:** the scoring formula and thresholds are design choices, not validated against real hiring outcomes.

## Roadmap

- RAG over a knowledge base of the candidate's projects and certificates, so evidence can cite specific documents
- An agent that collects job ads from official sources and evaluates them automatically
- Human-in-the-loop approval before any action, such as drafting an application