import json
from pathlib import Path

from scoring import compute_score, recommend

data = json.loads(Path("results.json").read_text(encoding="utf-8"))
results = data["results"]

for r in results:
    old_score = r["fit_score"]
    r["fit_score"] = compute_score(r["requirements"], r["score_cap"])
    r["recommendation"] = recommend(r["fit_score"])
    change = "" if old_score == r["fit_score"] else f"  (was {old_score})"
    print(f"{r['fit_score']:>2}/10  {r['recommendation']:<22}  {r['ad_file']}{change}")