import json
from pathlib import Path

import anthropic

MODEL = "claude-haiku-5-5"
MAX_STEPS = 5
SYSTEM_PROMPT = (
    "You help the user with their job search using their saved job evaluations. "
    "Always use the tools to look up scores and requirements; never invent them. "
    "Be concise."
)

client = anthropic.Anthropic()


# ---------- Tool functions ----------

def load_results():
    data = json.loads(Path("results.json").read_text(encoding="utf-8"))
    return data["results"]


def get_evaluations(min_score=1):
    if not isinstance(min_score, int) or not 1 <= min_score <= 10:
        raise ValueError("min_score must be an integer from 1 to 10")
    return [
        {
            "ad_file": r["ad_file"],
            "fit_score": r["fit_score"],
            "recommendation": r["recommendation"],
            "main_gap_to_close": r["main_gap_to_close"],
        }
        for r in load_results()
        if r["fit_score"] >= min_score
    ]


def get_evaluation_details(ad_file):
    results = load_results()
    for r in results:
        if r["ad_file"] == ad_file:
            return {
                "ad_file": r["ad_file"],
                "fit_score": r["fit_score"],
                "requirements": r["requirements"],
                "extra_strengths": r["extra_strengths"],
                "preference_check": r["preference_check"],
            }
    known = [r["ad_file"] for r in results]
    raise ValueError(f"Unknown ad_file '{ad_file}'. Known files: {known}")


TOOL_FUNCTIONS = {
    "get_evaluations": get_evaluations,
    "get_evaluation_details": get_evaluation_details,
}

tools = [
    {
        "name": "get_evaluations",
        "description": (
            "Returns a summary of the user's saved job evaluations: ad file name, "
            "fit score (1-10), recommendation, and main gap to close. "
            "Use this first to see which jobs exist. Can filter by minimum fit score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_score": {
                    "type": "integer",
                    "description": "Only return evaluations with at least this fit score (1-10).",
                }
            },
            "required": [],
        },
    },
    {
        "name": "get_evaluation_details",
        "description": (
            "Returns the full evaluation of ONE job: every requirement with its type "
            "and evidence level, extra strengths, and the preference check. "
            "Use this when the user asks about the details of a specific job."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "ad_file": {
                    "type": "string",
                    "description": "The ad file name exactly as returned by get_evaluations, e.g. 'ad3.txt'.",
                }
            },
            "required": ["ad_file"],
        },
    },
]


# ---------- Running tools safely ----------

def run_tool(name, tool_input):
    if name not in TOOL_FUNCTIONS:
        return f"Error: unknown tool '{name}'", True
    try:
        result = TOOL_FUNCTIONS[name](**tool_input)
        return json.dumps(result, ensure_ascii=False), False
    except Exception as error:
        return f"Error: {error}", True


# ---------- The agent loop ----------

def ask(question):
    messages = [{"role": "user", "content": question}]

    for step in range(1, MAX_STEPS + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=4000,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            return "".join(b.text for b in response.content if b.type == "text")

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                content, is_error = run_tool(block.name, block.input)
                status = "ERROR" if is_error else "ok"
                print(f"  [step {step}] {block.name}({block.input}) -> {status}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": content,
                    "is_error": is_error,
                })

        messages.append({"role": "user", "content": tool_results})

    return "Stopped: reached the maximum number of steps."



def main():
    print("Job assistant ready. Type 'quit' to exit.\n")
    while True:
        question = input("You: ")
        if question.strip().lower() == "quit":
            break
        if not question.strip():
            continue
        print(f"\nAssistant: {ask(question)}\n")


if __name__ == "__main__":
    main()