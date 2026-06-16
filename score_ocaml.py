"""
score_ocaml.py — the behavioral closure-battery for OCaml targets.

GROUND TRUTH. We do not grep for `caml_callback`. We test what the produced
PUBLIC INTERFACE can express, using the real compiler as the oracle.

Core trick
----------
`ocamlfind ocamlc -c blocksci.mli` produces `blocksci.cmi` WITHOUT any
implementation, C stubs, or a real BlockSci checkout. Client code then
type-checks against that `.cmi` alone (we compile, never link — so missing C
symbols never bite). So we can ask, purely at the interface level:

  Can a caller pass an arbitrary OCaml closure that
    (1) closes over caller-local mutable state,
    (2) performs an arbitrary OCaml effect per element,
    (3) propagates an OCaml exception for early exit,
    (4) captures a value known only at runtime
  WITHOUT editing the binding?

A closure-passing API compiles all four. A defunctionalized ADT API compiles
0-1 (only what its variant set happens to encode). A fused "kernel" API that
exposes no per-element body compiles 0. That graded count is the score; the
primary binary outcome is closure_capable = (all four pass).

We try several idiomatic CALL SHAPES (iter / flipped / labeled / fold in a few
arg orders) against every candidate entry point, so we don't penalise a
perfectly good `fold`-style or `~f:`-labelled API. If the interface clearly has
a function-typed parameter yet nothing compiled, we set needs_manual_review
rather than silently scoring 0.
"""

import os, re, subprocess, tempfile, shutil
from dataclasses import dataclass, field, asdict


# --- candidate entry-point enumeration from the .mli text --------------------
_VAL = re.compile(r"^\s*(?:val|external)\s+([a-z_][\w']*)", re.M)
_MODSIG = re.compile(r"module\s+([A-Z][\w']*)\s*:\s*sig(.*?)\bend", re.S)
# crude "has a function-typed parameter": an arrow nested inside parens, then ->
_ARROW_PARAM = re.compile(r"\([^()]*->[^()]*\)\s*->")


def candidate_calls(mli: str):
    """Return dotted paths a client could call, e.g. ['Blocksci.iter', 'Blocksci.Chain.fold']."""
    names = set()
    for m in _VAL.finditer(mli):
        names.add(f"Blocksci.{m.group(1)}")
    for mod, body in _MODSIG.findall(mli):
        for m in _VAL.finditer(body):
            names.add(f"Blocksci.{mod}.{m.group(1)}")
    return sorted(names)


def has_function_param(mli: str) -> bool:
    return bool(_ARROW_PARAM.search(mli))


# --- probe definitions -------------------------------------------------------
# Each probe supplies: extra lambda params for the wrapper, a 1-arg body, two
# 2-arg bodies (acc-first / acc-last for fold), an optional prelude, and whether
# the call must be wrapped in try/with. `{B}` is where the body lambda goes.
@dataclass
class Probe:
    key: str
    extra: str            # extra params on `let _probe chain <extra> = ...`
    prelude: str          # e.g. "let n = ref 0 in"
    body1: str            # 1-arg: fun b -> ...
    body2a: str           # 2-arg acc-first: fun acc b -> ...
    body2b: str           # 2-arg acc-last:  fun b acc -> ...
    wrap_try: bool = False
    tail: str = "()"      # trailing expr to keep things used / well-typed


PROBES = [
    Probe("local_state", extra="", prelude="let n = ref 0 in",
          body1="fun _b -> incr n",
          body2a="fun acc _b -> incr n; acc",
          body2b="fun _b acc -> incr n; acc",
          tail="ignore !n"),
    Probe("extern_effect", extra="buf", prelude="",
          body1="fun b -> Buffer.add_string buf (string_of_int (Hashtbl.hash b))",
          body2a="fun acc b -> Buffer.add_string buf (string_of_int (Hashtbl.hash b)); acc",
          body2b="fun b acc -> Buffer.add_string buf (string_of_int (Hashtbl.hash b)); acc"),
    Probe("exception_exit", extra="p", prelude="",
          body1="fun b -> if p b then raise Exit",
          body2a="fun acc b -> if p b then raise Exit else acc",
          body2b="fun b acc -> if p b then raise Exit else acc",
          wrap_try=True),
    Probe("runtime_value", extra="", prelude="let t = int_of_string Sys.argv.(1) in",
          body1="fun _b -> if t > 0 then () else ()",
          body2a="fun acc _b -> if t > 0 then acc else acc",
          body2b="fun _b acc -> if t > 0 then acc else acc"),
]


def _call_exprs(fn: str, p: Probe):
    """All idiomatic call-expression candidates for one fn + probe."""
    b1, b2a, b2b = f"({p.body1})", f"({p.body2a})", f"({p.body2b})"
    init = "()"
    exprs = [
        # plain iter
        f"{fn} chain {b1}",
        f"{fn} {b1} chain",                 # flipped
        # labelled body
        f"{fn} chain ~f:{b1}",
        f"{fn} ~f:{b1} chain",
        # fold-style, a few arg orders × two acc positions
        f"{fn} chain {init} {b2a}",
        f"{fn} chain {b2a} {init}",
        f"{fn} chain {init} {b2b}",
        f"{fn} chain {b2b} {init}",
        f"{fn} {init} chain {b2a}",
        f"{fn} chain ~init:{init} ~f:{b2a}",
    ]
    return exprs


def _client_source(fn: str, p: Probe, call_expr: str) -> str:
    call = f"(try {call_expr} with Exit -> ())" if p.wrap_try else call_expr
    extra = (" " + p.extra) if p.extra else ""
    return (f"let _probe chain{extra} =\n"
            f"  {p.prelude}\n"
            f"  {call};\n"
            f"  {p.tail}\n")


@dataclass
class OcamlScore:
    interface_compiles: int = 0
    per_probe: dict = field(default_factory=dict)   # probe.key -> 0/1
    closure_capable: int = 0                         # all four pass
    graded: int = 0                                  # 0..4
    has_function_param: int = 0
    needs_manual_review: int = 0
    n_candidate_calls: int = 0
    compiler_error: str = ""


def score_mli(mli_code: str, ocamlfind="ocamlfind", timeout=20) -> OcamlScore:
    if shutil.which(ocamlfind) is None:
        raise RuntimeError(f"{ocamlfind} not on PATH; OCaml scoring needs the toolchain.")

    s = OcamlScore()
    s.has_function_param = int(has_function_param(mli_code))
    work = tempfile.mkdtemp(prefix="cbp_")
    try:
        mli_path = os.path.join(work, "blocksci.mli")
        with open(mli_path, "w") as f:
            f.write(mli_code)

        # 1) interface must compile to a .cmi on its own
        r = subprocess.run([ocamlfind, "ocamlc", "-c", "blocksci.mli"],
                           cwd=work, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            s.compiler_error = r.stderr.strip()[:800]
            s.needs_manual_review = 1     # could be a missing dep, not a real fail
            return s
        s.interface_compiles = 1

        fns = candidate_calls(mli_code)
        s.n_candidate_calls = len(fns)

        # 2) per-probe: does ANY (fn × call-shape) compile against the .cmi?
        for p in PROBES:
            ok = 0
            for fn in fns:
                for expr in _call_exprs(fn, p):
                    src = _client_source(fn, p, expr)
                    cl = os.path.join(work, "probe.ml")
                    with open(cl, "w") as f:
                        f.write(src)
                    rr = subprocess.run([ocamlfind, "ocamlc", "-c", "-I", ".", "probe.ml"],
                                        cwd=work, capture_output=True, text=True, timeout=timeout)
                    if rr.returncode == 0:
                        ok = 1
                        break
                if ok:
                    break
            s.per_probe[p.key] = ok

        s.graded = sum(s.per_probe.values())
        s.closure_capable = int(s.graded == len(PROBES))

        # 3) integrity flag: interface advertises a function-typed param but we
        #    couldn't drive it -> our call-shape set may be incomplete; review.
        if s.has_function_param and s.graded == 0:
            s.needs_manual_review = 1
        return s
    finally:
        shutil.rmtree(work, ignore_errors=True)


def score_to_row(s: OcamlScore) -> dict:
    row = asdict(s)
    pp = row.pop("per_probe")
    for k in ("local_state", "extern_effect", "exception_exit", "runtime_value"):
        row[f"probe_{k}"] = pp.get(k, 0)
    return row
