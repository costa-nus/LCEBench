"""IFBench scoring: fraction of the prompt's instructions that the response follows (loose
matching as in the official evaluation) plus a textual feedback listing followed/violated ones."""

from .utils_ifbench import instructions_registry


def score_with_feedback(instruction_id_list, kwargs, prompt, response):
    """Return (fraction of instructions followed, feedback text).

    ``kwargs`` is the example's own list; None-valued arguments are filtered in place.
    """
    r = response.split("\n")
    response_remove_first = "\n".join(r[1:]).strip()
    response_remove_last = "\n".join(r[:-1]).strip()
    response_remove_both = "\n".join(r[1:-1]).strip()
    revised_response = response.replace("*", "")
    revised_response_remove_first = response_remove_first.replace("*", "")
    revised_response_remove_last = response_remove_last.replace("*", "")
    revised_response_remove_both = response_remove_both.replace("*", "")
    all_responses = [
        response,
        revised_response,
        response_remove_first,
        response_remove_last,
        response_remove_both,
        revised_response_remove_first,
        revised_response_remove_last,
        revised_response_remove_both,
    ]
    is_following_list = []

    correct_feedbacks = []
    incorrect_feedbacks = []

    for index, instruction_id in enumerate(instruction_id_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        kwargs[index] = {k: v for k, v in kwargs[index].items() if v is not None}

        ins_text = instruction.build_description(**kwargs[index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            ins_text = instruction.build_description(prompt=prompt)

        is_following = False
        for r in all_responses:
            if r.strip() and instruction.check_following(r):
                is_following = True
                break

        if not is_following:
            incorrect_feedbacks.append(ins_text)
        else:
            correct_feedbacks.append(ins_text)

        is_following_list.append(is_following)

    correct_feedback_text = ""
    if len(correct_feedbacks) > 0:
        correct_feedback_text = (
            "Your response correctly followed the following instructions:\n"
            + "\n".join(correct_feedbacks)
        )

    incorrect_feedback_text = ""
    if len(incorrect_feedbacks) > 0 and len(correct_feedbacks) > 0:
        incorrect_feedback_text = (
            "However, your response did not follow the following instructions properly:\n"
            + "\n".join(incorrect_feedbacks)
        )
    elif len(incorrect_feedbacks) > 0:
        incorrect_feedback_text = (
            "Your response did not follow the following instructions properly:\n"
            + "\n".join(incorrect_feedbacks)
        )

    feedback_text = correct_feedback_text + "\n" + incorrect_feedback_text
    feedback_text = feedback_text.strip()

    return sum(is_following_list) / len(is_following_list), feedback_text


def metric_with_feedback(item, response):
    """Raw-record interface (ACE, TF-GRPO): ``item`` is a dataset record, ``response`` a string."""
    return score_with_feedback(item["instruction_id_list"], item["kwargs"], item["prompt"], response)


def dspy_metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    """DSPy/GEPA interface: ``example`` is a dspy.Example, ``pred`` has a ``response`` field."""
    import dspy

    score, feedback = score_with_feedback(example.instruction_id_list, example.kwargs, example.prompt, pred.response)
    return dspy.Prediction(score=score, feedback=feedback)


def dspy_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    return dspy_metric_with_feedback(example, pred, trace).score
