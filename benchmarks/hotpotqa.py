"""HotpotQA helpers: answers are wrapped in <answer> ... </answer> and scored by exact match
after SQuAD-style normalisation. The gold and distractor paragraphs are given in the context."""

import re
import string
import unicodedata


def normalize_text(s):
    """Unicode NFD, lowercase, drop punctuation and articles, collapse whitespace."""
    s = unicodedata.normalize("NFD", s)

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction, ground_truth):
    return normalize_text(prediction) == normalize_text(ground_truth)


def textual_supporting_facts(context, supporting_facts):
    """Concatenate the gold supporting paragraphs as "title | sentences" (used as training feedback)."""
    title_to_sentences = {title: sentences for title, sentences in zip(context["title"], context["sentences"])}
    text = ""
    for title in set(supporting_facts["title"]):
        text += title + " | " + "".join(title_to_sentences[title])
    return text
