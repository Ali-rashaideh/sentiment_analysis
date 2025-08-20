import os, re
from pathlib import Path
from typing import Dict, List, Any
from transformers import pipeline


BASE = Path("./models").resolve()  # models path
WA, WB = 0.6, 0.4              # vote weights: primary model vs multilingual

# cache of loaded pipelines so we don't reload on every request
PIPES: Dict[str, Any] = {}

def load_pipe(which: str):
    """
    Load a local Transformers pipeline the first time we need it, then reuse it (Loading a model is slow and memory-heavy. we do it once to reuse many times).
    which: one of arabic,english,multi
    """
    if which not in PIPES:
        path = {"arabic": BASE/"arabic", "english": BASE/"english", "multi": BASE/"multi"}[which]
        PIPES[which] = pipeline(
            "sentiment-analysis",
            model=path.as_posix(),
            tokenizer=path.as_posix(),
            device=-1
        )
    return PIPES[which]

def normalize_label(x: str) -> str:
    """
    Map whatever the model returns into our 3 labels: negative, natural, positive.
    """
    t = x.strip().lower()
    if "neg" in t or t in {"label_0","0"}: return "negative"
    if "pos" in t or t in {"label_1","1"}: return "positive"
    if "neu" in t or "neutral" in t or t in {"label_2","2"}: return "natural"
    return "natural"

def ensure_three(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Make sure the scores dict has exactly our 3 keys and sums to 1.0.
    """
    s = {
        "positive": scores.get("positive", 0.0),
        "natural":  scores.get("natural",  0.0),
        "negative": scores.get("negative", 0.0),
    }
    total = s["positive"] + s["natural"] + s["negative"] or 1.0
    return {k: v/total for k, v in s.items()}

def arabic_checker(text: str) -> bool:
    """
    to see if there is arabic words to use the arabic model.
    """
    return bool(re.search(r"[\u0600-\u06FF]", text))

def predict_with(text: str, which: str) -> Dict[str, Any]:
    """
    Run model and return a dict: {model, label, scores}.
    scores is a normalized dict with the 3 labels as we did in normalize_label func.
    """
    pipe = load_pipe(which)
    out = pipe(text, truncation=True, top_k=None)  # truncation=True to handle long texts, top_k=None to get all labels

    # some pipeline versions return [[{...}]]; flatten that shape
    if isinstance(out, list) and out and isinstance(out[0], list):
        out = out[0]
    scores: Dict[str, float] = {}
    for item in out:
        scores[normalize_label(item["label"])] = float(item["score"])
    scores = ensure_three(scores)
    label = max(scores, key=lambda k: scores[k])
    return {"model": which, "label": label, "scores": scores}

def combine_scores(a: Dict[str, float], b: Dict[str, float], wa: float = WA, wb: float = WB) -> Dict[str, float]:
    """
    Weighted vote between two score dicts using the global weights WA (primary) and WB (multilingual).
    Returns a normalized dict so values sum to 1.0.
    """
    combined = {
        "positive": wa * a.get("positive", 0.0) + wb * b.get("positive", 0.0),
        "natural":  wa * a.get("natural",  0.0) + wb * b.get("natural",  0.0),
        "negative": wa * a.get("negative", 0.0) + wb * b.get("negative", 0.0),
    }
    total = sum(combined.values()) or 1.0
    return {k: v/total for k, v in combined.items()}

def analyse(text: str) -> Dict[str, Any]:
    """
    Full analysis for one sentence:
      - choose (arabic + multi) or (english + multi)
      - compute weighted vote for final label
      - report which single model was most confident in its own prediction
      - include confidence numbers for both the vote and best-model choice
    """
    primary = "arabic" if arabic_checker(text) else "english"
    m1 = predict_with(text, primary)     # primary model result
    m2 = predict_with(text, "multi")     # multilingual model result

    combined = combine_scores(m1["scores"], m2["scores"], WA, WB)
    final_label = max(combined, key=combined.get)
    final_confidence = combined[final_label]
    # margin = winner minus second-best -> quick certainty gauge
    sorted_vals = sorted(combined.values(), reverse=True)
    final_margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) >= 2 else final_confidence

    # "best model" = the single model most confident in its own top label
    m1_conf = m1["scores"][m1["label"]]
    m2_conf = m2["scores"][m2["label"]]
    best = m1 if m1_conf >= m2_conf else m2

    return {
        "sentence": text,
        "final_label": final_label,
        "final_confidence": final_confidence,
        "final_margin": final_margin,
        "vote_weights": {"primary": WA, "secondary": WB},   # now actually used
        "used_models": [m1["model"], m2["model"]],
        "best_model": best["model"],
        "best_model_confidence": best["scores"][best["label"]],
        "per_model": [m1, m2]
    }

def analyse_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """
    Apply analyse() to a list of sentences.
    """
    return [analyse(t) for t in texts]