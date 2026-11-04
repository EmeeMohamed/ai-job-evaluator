import anthropic

MODEL = "claude-haiku-5-5"
SYSTEM_PROMPT = "You are a helpful, concise assistant."

MAX_MESSAGES = 3

PRICES = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
}

client = anthropic.Anthropic()

messages = []

total_cost = 0.0
summary = ""

print("Chat started. Type 'quit' to exit.\n")

def summarize(old_summary, dropped_messages):
    transcript = "\n".join(
        f"{m['role']}: {m['content']}" for m in dropped_messages
    )
    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=(
           "You maintain a short memory of a conversation. "
            "Keep facts about the user, their goals, decisions made, "
            "and open questions. Drop small talk. Use short bullet points." 
        ),
        messages=[{
            "role": "user",
            "content": (
                f"<current_summary>\n{old_summary}\n</current_summary>\n\n"
                f"<new_messages>\n{transcript}\n</new_messages>\n\n"
                "Update the summary to include the important information "
                "from the new messages. Return only the updated summary."
            )
        }],
    )
    text = "".join(b.text for b in response.content if b.type == "text")
    price = PRICES[MODEL]
    cost = (response.usage.input_tokens * price["input"]
            + response.usage.output_tokens * price["output"]) / 1_000_000
    return text, cost

while True:
    user_input = input("You: ")

    if user_input.strip().lower() == "quit":
        break
    if not user_input.strip():
        continue

    messages.append({"role": "user", "content": user_input})

    if len(messages) > MAX_MESSAGES:
        dropped = messages[:-MAX_MESSAGES]
        messages = messages[-MAX_MESSAGES:]
        summary, summary_cost = summarize(summary, dropped)
        total_cost += summary_cost
        print(f"\n[memory updated]\n{summary}\n")
    while messages[0]["role"] != "user":
        messages.pop(0)

    system_text = SYSTEM_PROMPT
    if summary:
        system_text += f"\n\n<conversation_summary>\n{summary}\n</conversation_summary>"

    sent = len(messages)

    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=system_text,
        messages=messages,
    )

    reply = ""
    for block in response.content:
        if block.type == "text":
            reply += block.text

    if response.stop_reason == "max_tokens":
        print("\n[Warning: this reply was cut off by max_tokens]")

    messages.append({"role": "assistant", "content": reply})

    price = PRICES[MODEL]
    turn_cost = (response.usage.input_tokens * price["input"]
                 + response.usage.output_tokens * price["output"]) / 1_000_000
    total_cost += turn_cost

    print(f"\nClaude: {reply}\n")
    print(f"[messages sent: {sent} | input tokens: {response.usage.input_tokens} "
          f"| turn: ${turn_cost:.6f} | total: ${total_cost:.6f}]\n")
