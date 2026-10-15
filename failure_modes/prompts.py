import json

from .appworld_instructions import APPWORLD_INITIAL_INSTRUCTIONS


def _render(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _append_appworld_context(prompt, record):
    method = str(record.get("method") or "").strip().upper()
    if method not in APPWORLD_INITIAL_INSTRUCTIONS:
        raise ValueError(f"unsupported AppWorld method: {method!r}")
    for title, value in (
        ("Task context", record.get("task_context")),
        ("Task instructions", APPWORLD_INITIAL_INSTRUCTIONS[method]),
        ("Message history", record.get("history_messages")),
        ("Ground truth code", record.get("ground_truth_code")),
        ("Test report", record.get("test_report")),
        ("Feedback", record.get("feedback")),
    ):
        prompt += f"\n\n{title}:\n{_render(value)}"
    return prompt


def build_label_prompt(record):
    """Build the original concise prompt from one evaluated error trajectory."""
    prompt = """Diagnose the failure in this validated error trajectory.

Describe the smallest reusable failure mechanism that caused the error.

Requirements:
- Return exactly one short, specific sentence.
- Identify the failed operation, relation, constraint, evidence use or inference step.
- Use semantic roles instead of instance-specific names, dates, values, or positions.
- Do not add background, advice, headings, bullets, or generic evaluation.
- Do not state only the outcome, such as "incorrect answer" or "reasoning failure".
- Base the diagnosis only on the supplied trajectory and evaluator evidence.
- Failure may be caused by output that violates the task instructions, such as an unexpected format.
"""
    if record.get("dataset") == "appworld":
        prompt = _append_appworld_context(prompt, record)
        return prompt + (
            "\n\nReturn only JSON with exactly one field:\n"
            '{"failure_mechanism": "One short sentence describing the reusable failure mechanism."}'
        )
    prompt += f"\nQuestion:\n{record['question']}"
    for field, title in (
        ("task_instruction", "Task instruction used by the generator"),
        ("input_context", "Input context"),
        ("supporting_facts_text", "Gold supporting-facts passages for only diagnosing evidence use"),
    ):
        if record.get(field):
            prompt += f"\n\n{title}:\n{_render(record[field])}"
    if record.get("dataset") != "ifbench" and record.get("ground_truth") is not None:
        prompt += f"\n\nReference answer:\n{_render(record['ground_truth'])}"
    prompt += (
        f"\n\nComplete model reasoning or raw output:\n{_render(record['raw_output'])}"
        "\n\nExtracted prediction following the task instructions:\n"
        f"{_render(record['extracted_prediction'])}"
        f"\n\nScore:\n{_render(record['score'])}"
    )
    if record.get("dataset") == "ifbench" and record.get("feedback"):
        prompt += f"\n\nEvaluator feedback:\n{_render(record['feedback'])}"
    prompt += (
        "\n\nReturn only JSON with exactly one field:\n"
        '{"failure_mechanism": "One short sentence describing the reusable failure mechanism."}'
    )
    return prompt
