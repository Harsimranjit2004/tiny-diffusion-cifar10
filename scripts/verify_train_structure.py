"""
scripts/verify_train_structure.py

PHASE 3 — STRUCTURAL INTEGRITY CHECK FOR train.py

WHY THIS SCRIPT EXISTS: src/tiny_diffusion/training/train.py has been
broken by editing mistakes THREE separate times during this project —
each time a function's signature line was accidentally deleted while
inserting a new function nearby, leaving an orphaned docstring and body
silently merged into the PRECEDING function as dead/unreachable code.
Critically, all three times the file still PASSED a plain `ast.parse()`
syntax check, because the corruption was structurally valid Python (a
docstring sitting at the start of an unreachable code block looks
exactly like any other string literal statement to the parser) — syntax
validity alone was never enough to catch this class of bug.

This script checks something stronger than syntax: that every function
this file is KNOWN to need actually exists as an independently callable
top-level function, with the right parameters, not merged into another
function as dead code.

RUN THIS after ANY edit to train.py, before trusting the file:
    python scripts/verify_train_structure.py
"""

import ast
import sys
from pathlib import Path

# The complete list of functions train.py must define at module level,
# with their expected parameter names — update this list if you
# deliberately add/remove/rename a function.
EXPECTED_FUNCTIONS = {
    "build_model_config": ["cfg"],
    "compute_grad_norm": ["model"],
    "generate_sample_grid": [
        "model",
        "schedule",
        "ema",
        "num_classes",
        "image_size",
        "device",
        "normalize_mean",
        "normalize_std",
        "num_samples_per_class",
        "ddim_steps",
    ],
    "is_running_on_sagemaker": [],
    "get_checkpoint_dir": [],
    "find_latest_checkpoint": ["checkpoint_dir"],
    "load_checkpoint_for_resume": [
        "checkpoint_path",
        "model",
        "optimizer",
        "ema",
        "device",
    ],
    "train": ["cfg"],
}


def verify(filepath: str) -> bool:
    with open(filepath) as f:
        source = f.read()

    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        print(f"SYNTAX ERROR — file does not even parse: {e}")
        return False

    found_functions: dict = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            params = [a.arg for a in node.args.args]
            found_functions[node.name] = (params, node.lineno, node.end_lineno)

    all_ok = True
    print(f"Checking {filepath} against {len(EXPECTED_FUNCTIONS)} expected functions...\n")

    for name, expected_params in EXPECTED_FUNCTIONS.items():
        if name not in found_functions:
            print(
                f"  ✗ MISSING — '{name}' does not exist as a top-level function at all. "
                "This is exactly the bug class that broke this file 3 times before: "
                "a signature line was likely deleted during an edit."
            )
            all_ok = False
            continue

        actual_params, start, end = found_functions[name]
        if actual_params != expected_params:
            print(
                f"  ✗ PARAM MISMATCH — '{name}' (lines {start}-{end}) has params "
                f"{actual_params}, expected {expected_params}"
            )
            all_ok = False
            continue

        # Sanity check: function body shouldn't be suspiciously tiny relative
        # to what we know it should contain (catches the case where a function
        # IS found by this name, but it's actually a stub/fragment, not the
        # real implementation merged in from elsewhere).
        line_count = end - start
        if line_count < 1:
            print(
                f"  ✗ SUSPICIOUSLY EMPTY — '{name}' (lines {start}-{end}) spans "
                "0 lines, likely a broken/empty definition."
            )
            all_ok = False
            continue

        print(f"  ✓ OK — '{name}' (lines {start}-{end}, {line_count} lines)")

    print()
    # Module-level statement check: catches code that should be inside a
    # function but is sitting at module level instead (executes on import).
    module_level_types = [type(node).__name__ for node in tree.body]
    unexpected = [
        t
        for t in module_level_types
        if t
        not in (
            "Expr",
            "Import",
            "ImportFrom",
            "FunctionDef",
            "ClassDef",
            "Assign",
            "AnnAssign",
            "If",
        )
    ]
    if unexpected:
        print(
            f"  ✗ UNEXPECTED MODULE-LEVEL CODE: {unexpected} — this usually "
            "means a function body leaked out to module scope."
        )
        all_ok = False
    else:
        print(
            "  ✓ Module-level statements are all expected types "
            "(imports, function/class defs) — no leaked executable code."
        )

    return all_ok


if __name__ == "__main__":
    target = "src/tiny_diffusion/training/train.py"
    if len(sys.argv) > 1:
        target = sys.argv[1]

    if not Path(target).exists():
        print(f"File not found: {target}")
        sys.exit(1)

    print("=" * 70)
    ok = verify(target)
    print("=" * 70)

    if ok:
        print("ALL CHECKS PASSED — file structure is sound.")
        sys.exit(0)
    else:
        print("STRUCTURAL ISSUES FOUND — fix before trusting this file.")
        sys.exit(1)
