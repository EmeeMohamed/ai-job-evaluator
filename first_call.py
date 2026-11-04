import anthropic

# MODEL = "claude-haiku-5-5"
# MODEL = "claude-sonnet-5-5"
MODEL = "claude-opus-5-5" # opus is a thinker model, so I have to be sure that the content block have a thinking type

# price per 1000000 tokens in USD
PRICES = {
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00},
    "claude-opus-5-5": {"input": 4.00, "output": 20.00},
}


client = anthropic.Anthropic()

response = client.messages.create(
    model=MODEL,
    max_tokens=300,
    system="You are a patient teacher for 10-year-olds. Use one everyday example.",
    messages=[
        {"role": "user", "content": "Explain in two sentences what an API is."}
    ],
)

for block in response.content:
    if block.type == "text":
        print(block.text)

print("Block type:", block.type)

print("---")
print("Stop reason:", response.stop_reason)
print("Input tokens:", response.usage.input_tokens)
print("Output tokens:", response.usage.output_tokens)

# print("=== Full response ===")
# print(response.model_dump_json(indent=2))

price = PRICES[MODEL]
input_cost = response.usage.input_tokens / 1_000_000 * price["input"]
output_cost = response.usage.output_tokens / 1_000_000 * price["output"]
total_cost = input_cost + output_cost

print(f"Model: {MODEL}")
print(f"Input cost:  ${input_cost:.7f}")
print(f"Output cost: ${output_cost:.7f}")
print(f"Total cost:  ${total_cost:.7f}")