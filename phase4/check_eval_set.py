import json
from pathlib import Path

docs = {p.name: p.read_text(encoding="utf-8") for p in Path("phase4/docs").glob("*.txt")}
cases = json.loads(Path("phase4/retrieval_eval.json").read_text(encoding="utf-8"))

for case in cases:
    phrase = case["must_contain"]
    found = {name: text.count(phrase) for name, text in docs.items() if phrase in text}
    total = sum(found.values())

    if total == 0:
        status = "MISSING"
    elif total > 1:
        status = "NOT UNIQUE"
    else:
        status = "OK"
    print(f"{status:<11} {phrase!r:45} {found}")

print(f"\n{len(cases)} test cases checked")