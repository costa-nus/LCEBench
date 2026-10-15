"""Task definitions for GEPA: examples, the DSPy program to optimize, and the (feedback) metrics.

Signature fields, descriptions and initial instructions, the program classes and their predictor
attribute names are those of the paper runs; GEPA optimizes the instructions of the single
predictor of each program.
"""

import dspy
from dspy.evaluate import answer_exact_match

from benchmarks import extract_answer_tag, load_split
from benchmarks.hotpotqa import textual_supporting_facts
from benchmarks.ifbench.metric import dspy_metric, dspy_metric_with_feedback
from benchmarks.math500.qwen_verify import compute_score
from benchmarks.middlematch import exact_match as middlematch_exact_match
from benchmarks.middlematch import extract_boxed_text


# ---------------------------------------------------------------- MATH500

class CoTSignature(dspy.Signature):
    question = dspy.InputField()

    answer = dspy.OutputField(desc="Put your final answer within \\boxed{{}}.")


class MATH_CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(CoTSignature)

    def forward(self, question):
        prediction = self.generate_answer(question=question)
        return prediction


def math500_examples(split):
    return [
        dspy.Example(question=r["question"], reasoning=r["reasoning"], answer=r["answer"]).with_inputs("question")
        for r in load_split("math500", split)
    ]


def math500_metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    ans_match = compute_score(prediction.answer, example.answer)
    if ans_match:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{prediction.answer}' is correct.")
    else:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{prediction.answer}' is incorrect. The correct answer is: {example.answer}.")


def math500_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return compute_score(prediction.answer, example.answer)


# ---------------------------------------------------------------- HotpotQA (passages given in the context)

class HotpotQASignature(dspy.Signature):
    question = dspy.InputField()
    context = dspy.InputField()

    answer = dspy.OutputField(desc=r"Directly provide the answer inside <answer> and </answer>, without detailed illustrations. For example, <answer> Beijing </answer>.")


class RAG(dspy.Module):
    """Answers from the given paragraphs (no retrieval step)."""

    def __init__(self):
        super().__init__()
        self.generate_answer = dspy.ChainOfThought(HotpotQASignature)

    def forward(self, question, context_text):
        prediction = self.generate_answer(context=context_text, question=question)
        prediction.answer = extract_answer_tag(prediction.answer)
        return prediction


def hotpotqa_examples(split):
    examples = []
    for r in load_split("hotpotqa", split):
        data = {k: r[k] for k in ["id", "question", "answer", "type", "supporting_facts", "context"] if k in r}
        data["gold_titles"] = set(data["supporting_facts"]["title"])
        data["context_text"] = r["context_text"]  # "Title: ...\nContent: ..." paragraphs
        examples.append(dspy.Example(**data).with_inputs("question", "context_text"))
    return examples


def hotpotqa_metric_with_feedback(example, prediction, trace=None, pred_name=None, pred_trace=None):
    ans_match = answer_exact_match(example, prediction)
    textual_context = ""
    if hasattr(prediction, "feedback_text"):
        textual_context = prediction.feedback_text + "\n\n"
    textual_context += textual_supporting_facts(example["context"], example["supporting_facts"])

    if ans_match:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{prediction.answer}' is correct. Here's some additional context behind the answer:\n{textual_context}")
    else:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{prediction.answer}' is incorrect. The correct answer is: {example.answer}. Here's some context behind the answer, and how you could have reasoned to get the correct answer:\n{textual_context}")


def hotpotqa_metric(example, prediction, trace=None, pred_name=None, pred_trace=None):
    return answer_exact_match(example, prediction)


# ---------------------------------------------------------------- IFBench

class IFBenchSignature(dspy.Signature):
    """Respond to the query. Ensure the response is correct and adheres to the given constraints."""

    query = dspy.InputField()
    response = dspy.OutputField()


class IFBenchCoT2StageProgram(dspy.Module):
    def __init__(self):
        self.generate_response_module = dspy.ChainOfThought(IFBenchSignature)

    def forward(self, prompt: str):
        response = self.generate_response_module(query=prompt)
        return response


def ifbench_examples(split):
    return [dspy.Example(**r).with_inputs("prompt") for r in load_split("ifbench", split)]


# ---------------------------------------------------------------- MiddleMatch

class MiddleMatchSignature(dspy.Signature):

    query = dspy.InputField()
    response = dspy.OutputField(desc=r"Put the exact final name within \boxed{}, for example, \boxed{Michael Jackson}.")


class CoTProgram(dspy.Module):
    def __init__(self):
        self.generate_response_module = dspy.ChainOfThought(MiddleMatchSignature)

    def forward(self, question: str):
        response = self.generate_response_module(query=question)
        response.response = extract_boxed_text(response.response)
        return response


def middlematch_examples(split):
    return [dspy.Example(**r).with_inputs("question") for r in load_split("middlematch", split)]


def middlematch_metric_with_feedback(example, pred, trace=None, pred_name=None, pred_trace=None):
    response = pred.response
    ans_match = middlematch_exact_match(example.answer, response)
    if ans_match:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{response}' is correct.")
    else:
        return dspy.Prediction(score=ans_match, feedback=f"The provided answer, '{response}' is incorrect. The correct answer is: {example.answer}.")


def middlematch_metric(example, pred, trace=None, pred_name=None, pred_trace=None):
    return middlematch_metric_with_feedback(example, pred, trace).score


# task -> (examples(split), program class, eval metric, feedback metric used by GEPA)
TASKS = {
    "hotpotqa": (hotpotqa_examples, RAG, hotpotqa_metric, hotpotqa_metric_with_feedback),
    "ifbench": (ifbench_examples, IFBenchCoT2StageProgram, dspy_metric, dspy_metric_with_feedback),
    "math500": (math500_examples, MATH_CoT, math500_metric, math500_metric_with_feedback),
    "middlematch": (middlematch_examples, CoTProgram, middlematch_metric, middlematch_metric_with_feedback),
}
