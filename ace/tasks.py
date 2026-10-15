"""Task definitions for ACE: sample processing, answer scoring and feedback, and per-task settings."""

from benchmarks import extract_answer_tag, load_split
from benchmarks import hotpotqa as hotpotqa_bench
from benchmarks import middlematch as middlematch_bench
from benchmarks.ifbench.metric import metric_with_feedback as ifbench_metric_with_feedback
from benchmarks.math500.qwen_verify import compute_score

from .prompts import CURATOR_PROMPTS, GENERATOR_PROMPTS


class AceTask:
    name = None
    eval_steps = 50          # validate (and possibly update the best playbook) every N training steps
    save_steps = 50          # save an intermediate playbook every N training steps
    no_ground_truth = False  # IFBench: no reference answer; the reflector/curator work from checker feedback
    uses_context = False     # HotpotQA: the generator and the curator also see the paragraphs
    fractional = False       # IFBench: scores are fractions of followed instructions
    reference_key = "target"

    @property
    def generator_prompt(self):
        return GENERATOR_PROMPTS[self.name]

    @property
    def curator_prompts(self):
        return CURATOR_PROMPTS[self.name]

    def load(self, split):
        return self.process(load_split(self.name, split))

    def process(self, raw_data):
        return [{"question": item.get("question", ""), "target": item.get("answer", "")} for item in raw_data]

    def reference(self, task_dict):
        return task_dict.get(self.reference_key, "")

    def generator_kwargs(self, task_dict):
        return {}

    def question_context(self, task_dict):
        return None

    def score(self, answer, reference):
        raise NotImplementedError

    def check(self, answer, task_dict):
        """Score of an answer plus the environment feedback for the reflector
        (None: the reflector gets the default "matches / does not match ground truth" message)."""
        return self.score(answer, self.reference(task_dict)), None

    def evaluate_accuracy(self, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("Predictions and ground truths must have same length")
        return sum(self.score(p, r) for p, r in zip(predictions, references)) / len(predictions) if predictions else 0.0


class Math500(AceTask):
    name = "math500"

    def score(self, answer, reference):
        return compute_score(answer, reference)


class MiddleMatch(AceTask):
    name = "middlematch"

    def score(self, answer, reference):
        return middlematch_bench.exact_match(middlematch_bench.extract_boxed_text(answer), reference)


class HotpotQA(AceTask):
    name = "hotpotqa"
    eval_steps = 75
    save_steps = 75
    uses_context = True

    def process(self, raw_data):
        return [
            {
                "context": item.get("context_text", ""),
                "question": item.get("question", ""),
                "target": item.get("answer", ""),
                "context_dict": item.get("context", ""),
                "supporting_facts": item.get("supporting_facts", ""),
            }
            for item in raw_data
        ]

    def generator_kwargs(self, task_dict):
        return {"context": task_dict.get("context", "")}

    def question_context(self, task_dict):
        return task_dict.get("context", "")

    def score(self, answer, reference):
        return hotpotqa_bench.exact_match(extract_answer_tag(answer), reference)

    def check(self, answer, task_dict):
        # Feedback shows the gold supporting paragraphs.
        textual_context = hotpotqa_bench.textual_supporting_facts(
            task_dict.get("context_dict", ""), task_dict.get("supporting_facts", "")
        )
        target = self.reference(task_dict)
        extracted_pred = extract_answer_tag(answer)
        score = hotpotqa_bench.exact_match(extracted_pred, target)
        if score:
            feedback = f"The provided answer, '{extracted_pred}' is correct. Here's some additional context behind the answer:\n{textual_context}"
        else:
            feedback = f"The provided answer, '{extracted_pred}' is incorrect. The correct answer is: {target}. Here's some context behind the answer, and how you could have reasoned to get the correct answer:\n{textual_context}"
        return score, feedback


class IFBench(AceTask):
    name = "ifbench"
    eval_steps = 75
    save_steps = 25
    no_ground_truth = True
    fractional = True
    reference_key = "item"

    def process(self, raw_data):
        return [{"question": item.get("prompt", ""), "item": item} for item in raw_data]

    def score(self, answer, reference):
        return ifbench_metric_with_feedback(reference, answer)[0]

    def check(self, answer, task_dict):
        return ifbench_metric_with_feedback(task_dict.get("item", ""), answer)


TASKS = {task.name: task for task in (HotpotQA(), IFBench(), Math500(), MiddleMatch())}
