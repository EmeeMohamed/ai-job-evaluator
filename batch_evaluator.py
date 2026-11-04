import json
from pathlib import Path
import anthropic
from scoring import compute_score, recommend

MODEL = "claude-haiku-5-5"

client = anthropic.Anthropic()

template = Path("prompts/job_evaluator_system.txt").read_text(encoding="utf-8")

candidate_file = Path("candidate.txt")
if not candidate_file.exists():
    candidate_file = Path("candidate.example.txt")
print(f"Using candidate file: {candidate_file}")

candidate = candidate_file.read_text(encoding="utf-8")
system_prompt = template.replace("{{CANDIDATE}}", candidate)

def evaluate_ad(job_ad):
    if not job_ad.strip():
        raise ValueError("Ad file is empty")
    
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": job_ad,
            }
        ]
    )
    if response.stop_reason == "max_tokens":
        raise ValueError("Response was cut off by max_tokens")

    raw_text = "".join(b.text for b in response.content if b.type == "text")
    result = json.loads(raw_text)

    result["fit_score"] = compute_score(result["requirements"], result.get("score_cap"))
    result["recommendation"] = recommend(result["fit_score"])
    return result

results = []
failures = []

for ad_path in sorted(Path("job_ads").glob("*.txt")):
    print(f"Evaluating {ad_path.name}...")
    try:
        job_ad = ad_path.read_text(encoding="utf-8")
        result = evaluate_ad(job_ad)
        result["ad_file"] = ad_path.name
        results.append(result)
        print(f"    OK: {result['fit_score']}/10")
    except Exception as error:
        print(f"    FAILED: {type(error).__name__}: {error}")
        failures.append({"ad_file": ad_path.name, "error": str(error)})

results.sort(key=lambda r: r["fit_score"], reverse=True)

print("\n=== Ranking ===")
for r in results:
    print(f"{r['fit_score']:>2}/10  {r['recommendation']:<22}  {r['ad_file']}")

if failures:
    print(f"\n{len(failures)} ad(s) failed:")
    for f in failures:
        print(f"  - {f['ad_file']}: {f['error']}")

output = {"results": results, "failures": failures}
Path("results.json").write_text(
    json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("\nSaved to results.json")