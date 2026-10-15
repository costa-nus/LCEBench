"""Prompt templates of Training-Free GRPO.

The rollout-summary / group-critique / batch-update templates are the official TF-GRPO prompts.
HotpotQA additionally shows the gold supporting facts to the teacher, and IFBench shows the
instruction-checker feedback instead of a ground-truth answer.

All templates are byte-identical to the ones used in the paper runs (including trailing spaces):
do not reformat them.
"""

# ---------------------------------------------------------------- student (one per task)

STUDENT_TEMPLATES = {
    "math500": """Solve the following problem step by step. Put your final answer within \\boxed{{}}.
{problem}

When solving problems, you MUST first carefully read and understand the helpful instructions and experiences:

{experiences}
""",
    "middlematch": """Solve the following problem step by step. Put the exact final name within \\boxed{{}}, for example, \\boxed{{Michael Jackson}}.
{problem}

When solving problems, you MUST first carefully read and understand the helpful instructions and experiences:

{experiences}
""",
    "hotpotqa": """Solve the following problem step by step. Directly provide the answer inside <answer> and </answer>, without detailed illustrations, for example, <answer> Beijing </answer>.

{problem}

{context}

When solving problems, you MUST first carefully read and understand the helpful instructions and experiences:

{experiences}
""",
    "ifbench": """Solve the following problem step by step. Respond to the query and provide the answer inside <answer> and </answer>, ensure the response is correct and adheres to the given constraints.
{problem}

When solving problems, you MUST first carefully read and understand the helpful instructions and experiences:

{experiences}
""",
}

# ---------------------------------------------------------------- teacher (ground-truth version)

SINGLE_ROLLOUT_SUMMARY_TEMPLATE = """An agent system may be provided with some experiences, and then it produces the following trajectory to solve the given problem. Please summarize the trajectory step-by-step:

1. For each step, describe **what action is being taken**, and which experience has been used in this step.
2. Given the grading of this rollout and the feedback, identify and explain any steps that **represent detours, errors, or backtracking**, highlighting why they might have occurred and what their impact was on the trajectory's progress. 
3. Maintain **all the core outcome of each step**, even if it was part of a flawed process.

<trajectory>
{trajectory}
</trajectory>

<evaluation>
{grade}
</evaluation>

<groundtruth>
{groundtruth}
</groundtruth>

Only return the trajectory summary of each step, e.g.,
1. what happened in the first step and the core outcomes
2. what happened in the second step and the core outcomes
3. ..."""


SINGLE_QUERY_CRITIQUE_TEMPLATE = """An agent system is provided with a set of experiences and has tried to solve the problem multiple times with both successful and wrong solutions. Review these problem-solving attempt and extract generalizable experiences. Follow these steps:

1. Trajectory Analysis:
    - For successful steps: Identify key correct decisions and insights
    - For errors: Pinpoint where and why the reasoning went wrong
    - Note any important patterns or strategies used/missed
    - Review why some trajectories fail? Is there any existing experiences are missed, or experiences do not provide enough guidance?

2. Update Existing Experiences
    - Some trajectories may be correct and others may be wrong, you should ensure there are experiences can help to run correctly
    - You have two options: [modify, add]
        * modify: You can modify current experiences to make it helpful
        * add: You can introduce new experiences may need to be 
    - You can update at most {max_operations} clear, generalizable lessons for this case
    - Before updating each experience, you need to:
        * Specify when it would be most relevant
        * List key problem features that make this experience applicable
        * Identify similar problem patterns where this advice applies
    
3. Requirements for each experience that is modified or added.
    - Begin with general background with several words in the experience
    - Focus on strategic thinking patterns, not specific calculations
    - Emphasize decision points that could apply to similar problems

Please provide reasoning in details under the guidance of the above 3 steps.
After the step-by-step reasoning, you will finish by returning in this JSON format as follows:
```json
[
    {{
        "option": "modify",
        "experience": "the modified experience",
        "modified_from": "G17" # specify the ID of experience that is modified
    }},
    {{
        "option": "add",
        "experience": "the added experience",
    }},
    ...
]
```
Note that your updated experiences may not need to cover all two options. Only using one type of updates is also very good.

<problem> 
{problem}
</problem>

<trajectories>
{trajectories}
</trajectories>

<groundtruth>
{groundtruth}
</groundtruth>

<experience>
{experiences}
</experience>"""


BATCH_EXPERIENCE_UPDATE_TEMPLATE = """An agent system is provided with a set of experiences and has tried to solve the problem multiple times. From the reflections, some suggestions on the existing experiences have been posed. Your task is to collect and think for the final experience revision plan. Each final experience must satisfy the following requirements.
1. It must be clear, generalizable lessons for this case, with no more than 32 words
2. Begin with general background with several words in the experience
3. Focus on strategic thinking patterns, not specific calculations
4. Emphasize decision points that could apply to similar problems
5. Avoid repeating saying similar experience in multiple different experiences

<existing_experiences> 
{experiences}
</existing_experiences>

<suggested_updates>
{updates}
</suggested_updates>

Please provide reasoning in each of the suggestions, and think for how to update existing experiences 
You have two update options: [modify, merge]
* modify: You can modify current experiences to make it helpful
* merge: You can merge some similar experiences into a more general forms to reduce duplication

After generating the step-by-step reasoning, you need to give the final experience revision details by returning in this JSON format as follows:
```json
[
    {{
        "option": "modify",
        "experience": "the modified experience",
        "modified_from": "C1" # specify the str ID of experience that is modified
    }},
    {{
        "option": "merge",
        "experience": "the merged experience",
        "merged_from": ["C1", "C3", "S4", ...] # specify the str IDs of experiences that is merged from, at least 2 IDs are needed
    }},
    ...
]
```

Your updated experiences may not need to cover all two options. Only using one type of updates is OK."""


# ---------------------------------------------------------------- task-specific teacher variants

def _replace_once(template, old, new):
    assert template.count(old) == 1, f"template variant anchor not found exactly once: {old!r}"
    return template.replace(old, new)


_GT_BLOCK = "<groundtruth>\n{groundtruth}\n</groundtruth>"
_HOTPOTQA_GT_BLOCK = "<groundtruth>\n{answer}\n</groundtruth>\n\n<supportingfacts>\n{supportingfacts}\n</supportingfacts>"
_IFBENCH_FEEDBACK_BLOCK = "<feedback>\n{feedback}\n</feedback>"

# task -> (rollout-summary template, group-critique template)
TEACHER_TEMPLATES = {
    "math500": (SINGLE_ROLLOUT_SUMMARY_TEMPLATE, SINGLE_QUERY_CRITIQUE_TEMPLATE),
    "middlematch": (SINGLE_ROLLOUT_SUMMARY_TEMPLATE, SINGLE_QUERY_CRITIQUE_TEMPLATE),
    "hotpotqa": (
        _replace_once(
            _replace_once(SINGLE_ROLLOUT_SUMMARY_TEMPLATE, "rollout and the feedback,", "rollout and the correct answer,"),
            _GT_BLOCK,
            _HOTPOTQA_GT_BLOCK,
        ),
        _replace_once(SINGLE_QUERY_CRITIQUE_TEMPLATE, _GT_BLOCK, _HOTPOTQA_GT_BLOCK),
    ),
    "ifbench": (
        _replace_once(SINGLE_ROLLOUT_SUMMARY_TEMPLATE, _GT_BLOCK, _IFBENCH_FEEDBACK_BLOCK),
        _replace_once(SINGLE_QUERY_CRITIQUE_TEMPLATE, _GT_BLOCK, _IFBENCH_FEEDBACK_BLOCK),
    ),
}
