from job_assistant import run_tool

test_cases = [
    ("get_evaluations", {"min_score": 15}),
    ("get_evaluations", {"min_score": "high"}),
    ("get_evaluations", {"limit": 3}),
    ("get_evaluation_details", {"ad_file": "../../Documents/private.txt"}),
    ("delete_all_files", {}),
    ("get_evaluations", {"min_score": 6}),
]

for name, tool_input in test_cases:
    content, is_error = run_tool(name, tool_input)
    status = "ERROR" if is_error else "OK   "
    print(f"{status}  {name}({tool_input})")
    print(f"       {content[:100]}\n")