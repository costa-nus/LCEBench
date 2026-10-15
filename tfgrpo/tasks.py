"""Task specifications for Training-Free GRPO.

Each task defines how a split is loaded, which fields fill the student template, how a response
is rewarded, what a rollout record stores, and which fields the teacher templates receive.
"""

from benchmarks import extract_answer_tag, load_split
from benchmarks import hotpotqa as hotpotqa_bench
from benchmarks import middlematch as middlematch_bench
from benchmarks.ifbench.metric import metric_with_feedback as ifbench_metric_with_feedback
from benchmarks.math500.qwen_verify import compute_score

from .prompts import STUDENT_TEMPLATES, TEACHER_TEMPLATES


def _rollout(sample_fields, reward, response, user_prompt, extra=None):
    record = {"problem": sample_fields["question"], **{k: v for k, v in sample_fields.items() if k != "question"}}
    record["reward"] = reward
    record.update(extra or {})
    record["trajectories"] = [{"trajectory": response}]
    record["user_prompt"] = user_prompt
    return record


class GroundTruthTask:
    """Tasks with a short reference answer (MATH500, MiddleMatch)."""

    def __init__(self, name, tag):
        self.name = name
        self.tag = tag
        self.student_template = STUDENT_TEMPLATES[name]
        self.summary_template, self.critique_template = TEACHER_TEMPLATES[name]

    def load(self, split):
        return [{"question": r.get("question", ""), "ground_truth": r.get("answer", "")} for r in load_split(self.name, split)]

    def student_fields(self, sample):
        return {"problem": sample["question"]}

    def score(self, response, sample):
        raise NotImplementedError

    def rollout(self, sample, response, user_prompt):
        fields = {"question": sample["question"], "ground_truth": sample["ground_truth"]}
        return _rollout(fields, self.score(response, sample), response, user_prompt)

    def teacher_fields(self, rollout):
        return {"groundtruth": rollout["ground_truth"]}


class Math500(GroundTruthTask):
    def score(self, response, sample):
        return compute_score(response, sample["ground_truth"])


class MiddleMatch(GroundTruthTask):
    def score(self, response, sample):
        return middlematch_bench.exact_match(middlematch_bench.extract_boxed_text(response), sample["ground_truth"])


class HotpotQA:
    name, tag = "hotpotqa", "HotpotQA"

    def __init__(self):
        self.student_template = STUDENT_TEMPLATES[self.name]
        self.summary_template, self.critique_template = TEACHER_TEMPLATES[self.name]

    def load(self, split):
        # The teacher sees the gold supporting paragraphs for training questions only.
        return [
            {
                "context": r.get("context_text", ""),
                "question": r.get("question", ""),
                "answer": r.get("answer", ""),
                "supporting_facts": (
                    hotpotqa_bench.textual_supporting_facts(r["context"], r["supporting_facts"]) if split == "train" else ""
                ),
            }
            for r in load_split(self.name, split)
        ]

    def student_fields(self, sample):
        return {"problem": sample["question"], "context": sample["context"]}

    def score(self, response, sample):
        return hotpotqa_bench.exact_match(extract_answer_tag(response), sample["answer"])

    def rollout(self, sample, response, user_prompt):
        fields = {
            "question": sample["question"],
            "groundtruth": sample["answer"],
            "context": sample["context"],
            "supporting_facts": sample["supporting_facts"],
        }
        return _rollout(fields, self.score(response, sample), response, user_prompt)

    def teacher_fields(self, rollout):
        return {"answer": rollout["groundtruth"], "context": rollout["context"], "supportingfacts": rollout["supporting_facts"]}


class IFBench:
    name, tag = "ifbench", "IFBench"

    def __init__(self):
        self.student_template = STUDENT_TEMPLATES[self.name]
        self.summary_template, self.critique_template = TEACHER_TEMPLATES[self.name]

    def load(self, split):
        return [{"question": r.get("prompt", ""), "item": r} for r in load_split(self.name, split)]

    def student_fields(self, sample):
        return {"problem": sample["question"]}

    @staticmethod
    def score_with_feedback(response, sample):
        """Fraction of instructions followed + checker feedback (the teacher sees the feedback)."""
        try:
            return ifbench_metric_with_feedback(sample["item"], extract_answer_tag(response, strip_fallback=False))
        except Exception:
            return 0, "Your response did not follow the instructions properly."

    def score(self, response, sample):
        return self.score_with_feedback(response, sample)[0]

    def rollout(self, sample, response, user_prompt):
        score, feedback = self.score_with_feedback(response, sample)
        fields = {"question": sample["question"], "item": sample["item"]}
        return _rollout(fields, score, response, user_prompt, extra={"feedback": feedback})

    def teacher_fields(self, rollout):
        return {"feedback": rollout["feedback"]}


TASKS = {
    "hotpotqa": HotpotQA(),
    "ifbench": IFBench(),
    "math500": Math500("math500", "MATH500"),
    "middlematch": MiddleMatch("middlematch", "MiddleMatch"),
}
