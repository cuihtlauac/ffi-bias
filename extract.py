"""
extract.py — pull the relevant code artifact out of a model completion.

We only need the language-facing interface to score capability:
  * OCaml  -> the .mli (interface). We compile *clients* against its .cmi.
  * others -> the public API surface (header / def block), fed to the LLM judge.

Heuristics, in order of preference:
  1. fenced blocks tagged with the language (```ocaml / ```ml / ```python / ...)
  2. for OCaml: prefer a block that looks like an interface (has `val`/`external`/`sig`)
  3. fall back to the largest fenced block, else the whole text.
"""

import re

FENCE = re.compile(r"```([\w+-]*)\n(.*?)```", re.DOTALL)

LANG_TAGS = {
    "ocaml": {"ocaml", "ml", "mli", "ocaml-interface"},
    "python": {"python", "py"},
    "julia": {"julia", "jl"},
    "lua": {"lua"},
    "c": {"c", "h", "cpp", "c++"},
}


def _blocks(text):
    return [(tag.lower().strip(), body) for tag, body in FENCE.findall(text)]


def _looks_like_mli(body: str) -> bool:
    return bool(re.search(r"^\s*(val|external)\s", body, re.M)) or "sig" in body


def extract_interface(text: str, lang: str) -> dict:
    """Return {'code': str, 'kind': 'tagged'|'untagged'|'fulltext', 'n_blocks': int}."""
    blocks = _blocks(text)
    tags = LANG_TAGS.get(lang, {lang})

    tagged = [b for (t, b) in blocks if t in tags]
    if lang == "ocaml":
        ifaces = [b for b in tagged if _looks_like_mli(b)]
        if ifaces:
            return dict(code=max(ifaces, key=len), kind="tagged", n_blocks=len(blocks))
    if tagged:
        return dict(code=max(tagged, key=len), kind="tagged", n_blocks=len(blocks))

    if blocks:  # untagged fences
        return dict(code=max((b for _, b in blocks), key=len),
                    kind="untagged", n_blocks=len(blocks))

    return dict(code=text, kind="fulltext", n_blocks=0)
