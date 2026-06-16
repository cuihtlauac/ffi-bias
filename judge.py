"""
judge.py — classify a produced API into the three patterns, for languages where
we have no portable compiler oracle (Python/Julia/Lua/C).

This is the WEAKER instrument. Use it only off the OCaml spine. To make it
credible we (a) run it on the OCaml samples too and (b) report Cohen's kappa
between judge and the real compiler (see analyze.py). If kappa is low, the
cross-language judge results are not trustworthy and you fall back to
human coding of a subsample.

The judge is deliberately blinded to the experimental cue: it sees only the
produced code + the language, never the prompt that generated it.
"""

import json, re
from api_client import call_model_meta

RUBRIC = """You are classifying the DESIGN of a language binding's per-element API.
You are given a target language and the produced public interface/binding code.

Decide which ONE pattern best describes how user code runs per element:

  A) CLOSURE_PASSING — the user passes a real {lang} function/closure/callback
     that the iterator invokes per element. The body is a first-class function
     of the host language, able to close over arbitrary local state and effects.

  B) DATA_AS_BEHAVIOR — the user passes a DATA description of the work (an
     expression tree, an ADT/variant, an enum of ops, a query/spec object) that
     the native side interprets. No host-language function is invoked per element.

  C) FUSED_KERNEL — the iteration and the body are fused on the native side and
     only the fused operation is exposed (e.g. a single "kernel"/"reduce"/
     "compute_stats" entry). The user cannot supply an arbitrary per-element body.

Also judge CLOSURE_CAPABLE: could a user, WITHOUT editing the binding, run an
arbitrary host-language body per element that closes over local mutable state,
calls unrelated host libraries, and can early-exit via an exception/break?
(A => yes; B and C => no.)

Return ONLY minified JSON, no prose, no markdown:
{"pattern":"A|B|C","closure_capable":true|false,"confidence":0.0-1.0,"why":"<=20 words"}
"""

_JSON = re.compile(r"\{.*\}", re.S)


def judge_api(code: str, lang: str, model: str) -> dict:
    # .replace, not .format: the rubric embeds a literal JSON example whose braces
    # would make str.format raise (KeyError on '"pattern"'). Only {lang} is a slot.
    sys = RUBRIC.replace("{lang}", lang)
    user = f"TARGET LANGUAGE: {lang}\n\nPRODUCED INTERFACE/BINDING:\n```\n{code[:12000]}\n```"
    # No temperature: Opus 4.8/4.7 reject it (see api_client.call_model). The judge
    # leans on a tightly-constrained rubric + minified-JSON output for stability
    # rather than temperature=0.
    txt, meta = call_model_meta(model=model, system=sys, user=user, max_tokens=300)
    usage = {"judge_input_tokens": meta.get("input_tokens"),
             "judge_output_tokens": meta.get("output_tokens")}
    m = _JSON.search(txt)
    if not m:
        return dict(pattern="?", closure_capable=None, confidence=0.0,
                    why="unparseable", raw=txt[:200], **usage)
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return dict(pattern="?", closure_capable=None, confidence=0.0,
                    why="bad json", raw=txt[:200], **usage)
    d["closure_capable"] = bool(d.get("closure_capable"))
    d.update(usage)
    return d


def judge_to_row(d: dict) -> dict:
    return {
        "judge_pattern": d.get("pattern", "?"),
        "judge_closure_capable": int(bool(d.get("closure_capable"))),
        "judge_confidence": float(d.get("confidence") or 0.0),
        "judge_why": d.get("why", "")[:120],
        "judge_input_tokens": d.get("judge_input_tokens"),
        "judge_output_tokens": d.get("judge_output_tokens"),
    }
