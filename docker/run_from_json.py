import ast
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import nbformat

NOTEBOOK_IN = "/app/notebook.ipynb"

# Fallback regex for cells that contain notebook magics (!, %, %%), which break ast.parse().
# This only patches simple single-line assignments like: param_x = something
_PARAM_ASSIGN_RE = re.compile(
    r"""^(?P<indent>\s*)
        (?P<name>param_[A-Za-z_]\w*)
        (?P<ann>\s*:\s*[^=]+)?          # optional type annotation
        \s*=\s*
        (?P<rhs>.*?)
        (?P<comment>\s*\#.*)?$
    """,
    re.VERBOSE | re.MULTILINE,
)


def load_payload() -> dict:
    # 1) JSON passed as first CLI argument OR a path to a json file
    if len(sys.argv) >= 2 and sys.argv[1].strip():
        arg = sys.argv[1].strip()
        if arg.endswith(".json") or os.path.exists(arg):
            return json.loads(Path(arg).read_text(encoding="utf-8-sig"))
        return json.loads(arg)

    # 2) Full payload JSON in env var
    if os.getenv("JOB_JSON"):
        return json.loads(os.environ["JOB_JSON"])

    # 3) AWS Batch style: NOTEBOOK_PARAMS contains just the params dict
    if os.getenv("NOTEBOOK_PARAMS"):
        try:
            params = json.loads(os.environ["NOTEBOOK_PARAMS"])
        except json.JSONDecodeError as e:
            raise SystemExit(f"NOTEBOOK_PARAMS is not valid JSON: {e}") from e

        payload = {"params": params}

        # output path is best provided by Batch as NOTEBOOK_OUT (S3 or local)
        payload["notebook_out"] = os.getenv("NOTEBOOK_OUT") or "/app/out/run.executed.ipynb"

        # Optional: keep job id around for logging / later use
        if os.getenv("JOB_ID"):
            payload["job_id"] = os.environ["JOB_ID"]

        return payload

    # 4) JSON file path (fallback)
    job_file = os.getenv("JOB_JSON_FILE", "/app/job.json")
    return json.loads(Path(job_file).read_text(encoding="utf-8-sig"))


def extract_params(payload: dict) -> dict:
    params = payload.get("params")
    if params is None:
        params = {k: v for k, v in payload.items() if k.startswith("param_")}
    else:
        params = {k: v for k, v in params.items() if k.startswith("param_")}

    if not params:
        raise SystemExit("No param_* values found in payload.")

    for k in params:
        if not k.isidentifier():
            raise SystemExit(f"Invalid parameter name: {k}")

    return params


def _line_start_offsets(text: str) -> list[int]:
    lines = text.splitlines(keepends=True)
    offsets = [0]
    cur = 0
    for ln in lines:
        offsets.append(cur)
        cur += len(ln)
    return offsets


def patch_cell_source(source: str, params: dict) -> tuple[str, int]:
    """
    Patch ALL occurrences of param_* assignments in a cell.

    - Preferred: AST patching (handles multiline RHS safely).
    - Fallback: regex line patching if AST fails (cells with %, !, %% magics).
    """
    # 1) AST patching
    try:
        tree = ast.parse(source)
        line_offsets = _line_start_offsets(source)
        patches = []

        def add_value_patch(value_node, var_name: str):
            if var_name not in params:
                return
            v = value_node
            if v is None:
                return
            if not all(hasattr(v, a) for a in ("lineno", "col_offset", "end_lineno", "end_col_offset")):
                return
            start = line_offsets[v.lineno] + v.col_offset
            end = line_offsets[v.end_lineno] + v.end_col_offset
            patches.append((start, end, repr(params[var_name])))

        for n in ast.walk(tree):
            if isinstance(n, ast.Assign):
                # Patch any assign where at least one target is a param_* we know about.
                param_targets = [
                    t.id
                    for t in n.targets
                    if isinstance(t, ast.Name) and t.id.startswith("param_") and t.id in params
                ]
                if param_targets:
                    add_value_patch(n.value, param_targets[-1])

            elif isinstance(n, ast.AnnAssign):
                if isinstance(n.target, ast.Name):
                    name = n.target.id
                    if name.startswith("param_") and name in params:
                        add_value_patch(n.value, name)

        if patches:
            patches.sort(key=lambda x: x[0], reverse=True)
            new_source = source
            for start, end, repl in patches:
                new_source = new_source[:start] + repl + new_source[end:]
            return new_source, len(patches)

    except SyntaxError:
        pass

    # 2) Regex fallback patching (single-line assignments)
    replaced = 0

    def repl(m: re.Match) -> str:
        nonlocal replaced
        name = m.group("name")
        if name in params:
            replaced += 1
            indent = m.group("indent") or ""
            ann = m.group("ann") or ""
            comment = m.group("comment") or ""
            return f"{indent}{name}{ann} = {repr(params[name])}{comment}"
        return m.group(0)

    new_source = _PARAM_ASSIGN_RE.sub(repl, source)
    return new_source, replaced


def main() -> int:
    payload = load_payload()

    notebook_out = payload.get("notebook_out") or "/app/out/run.executed.ipynb"
    params = extract_params(payload)

    nb = nbformat.read(NOTEBOOK_IN, as_version=4)

    total = 0
    for cell in nb.cells:
        if cell.get("cell_type") != "code":
            continue
        new_src, n = patch_cell_source(cell.source, params)
        if n:
            cell.source = new_src
            total += n

    if total == 0:
        raise SystemExit("No param_* assignments found to patch in the notebook.")

    with tempfile.TemporaryDirectory() as td:
        patched = Path(td) / "patched.ipynb"
        nbformat.write(nb, str(patched))

        cmd = ["papermill", str(patched), notebook_out]
        # Helpful in Batch logs
        if "job_id" in payload:
            print(f"JOB_ID={payload['job_id']}", flush=True)

        print(f"Patched {total} assignment(s).", flush=True)
        print("Running:", " ".join(cmd), flush=True)
        return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())