import json
from pathlib import Path
import anthropic

MODEL = "claude-haiku-5-5"
client = anthropic.Anthropic()

def get_evaluations(min_score=1):
    data = json.loads(Path("results.json").read_text(encoding="utf-8"))
    return [
        {
            "ad_file": r["ad_file"],
            "fit_score": r["fit_score"],
            "recommendation": r["recommendation"],
            "main_gap_to_close": r["main_gap_to_close"],
        }
        for r in data["results"]
        if r["fit_score"] >= min_score
    ]

tools = [
    {
        "name": "get_evaluations",
        "description": (
            "Returns the user's saved job ad evaluations: ad file name, fit score "
            "(1-10), recommendation, and the main gap to close. Use this whenever "
            "the user asks about their evaluated jobs. Can filter by minimum fit score."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "min_score": {
                    "type": "integer",
                    "description": "Only return evaluations with at least this fit score (1-10). Use 1 for all.",
                }
            },
            "required": [],
        },
    }
]

question = "Which jobs should I apply to, and what skill gap appears most often?"
#question = "Show me only the jobs that scored 6 or higher."
#question = "What is RAG, in one sentence?"
messages = [{"role": "user", "content": question}]

response = client.messages.create(
    model=MODEL, max_tokens=4000, tools=tools, messages=messages
)

print("Call 1 stop reason:", response.stop_reason)
print("Call 1 block types:", [b.type for b in response.content])

if response.stop_reason == "tool_use":
    messages.append({"role": "assistant", "content": response.content})

    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            print(f"\nClaude requests: {block.name}({block.input})")
            result = get_evaluations(**block.input)
            print(f"Our code returned {len(result)} evaluations")
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": json.dumps(result),
            })

    messages.append({"role": "user", "content": tool_results})

    # --- Call 2: Claude answers using the tool result ---
    response = client.messages.create(
        model=MODEL, max_tokens=4000, tools=tools, messages=messages
    )
    print("\nCall 2 stop reason:", response.stop_reason)

for block in response.content:
    if block.type == "text":
        print("\nClaude:", block.text)