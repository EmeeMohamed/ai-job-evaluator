import json
from pathlib import Path

from search import search

cases = json.loads(Path("phase4/retrieval_eval.json").read_text(encoding="utf-8"))

not_in_docs = [
    "What is my favorite food?",
    "Have I worked with Docker?",
    "Do I have a driving license?",
    "What salary do I expect?",
    "Habe ich schon bei Google gearbeitet?",
]

print("Score of the CORRECT chunk (answer is in the documents):")
correct_scores = []
for case in cases:
    hits = search(case["question"], k=10)
    score = next((h["similarity"] for h in hits if case["must_contain"] in h["text"]), None)
    correct_scores.append(score)
    print(f"  {score:.3f}  {case['question']}" if score is not None else f"  none   {case['question']}")

print("\nBEST score (answer is NOT in the documents):")
noise_scores = []
for question in not_in_docs:
    best = search(question, k=1)[0]["similarity"]
    noise_scores.append(best)
    print(f"  {best:.3f}  {question}")

found = [s for s in correct_scores if s is not None]
print(f"\nLowest correct score: {min(found):.3f}")
print(f"Highest noise score:  {max(noise_scores):.3f}")