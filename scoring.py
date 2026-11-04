EVIDENCE_POINTS = {"STRONG": 1.0, "WEAK": 0.5, "IN_PROGRESS": 0.25, "NONE": 0.0}

def coverage(items):
    if not items:
        return 1.0
    return sum(EVIDENCE_POINTS[r["evidence_level"]] for r in items) / len(items)


def compute_score(requirements, score_cap):
    required = [r for r in requirements if r["type"] == "REQUIRED"]
    nice = [r for r in requirements if r["type"] == "NICE_TO_HAVE"]

    if nice:
        raw_score = 1 + 7 * coverage(required) + 2 * coverage(nice)
    else:
        raw_score = 1 + 9 * coverage(required)

    score = round(raw_score)

    missing_required = sum(1 for r in required if r["evidence_level"] == "NONE")
    if missing_required >= 3:
        score = min(score, 4)
    elif missing_required >= 1:
        score = min(score, 6)

    if score_cap is not None:
        score = min(score, score_cap)
    return score


def recommend(score):
    if score >= 7:
        return "APPLY"
    if score >= 5:
        return "APPLY_AFTER_IMPROVING"
    return "SKIP"

