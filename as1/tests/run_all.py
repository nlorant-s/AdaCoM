#!/usr/bin/env python
"""Run every as1 unit test (CPU-only, no vLLM/Ray/HF needed).

    python as1/tests/run_all.py
"""
import os
import runpy
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))


def main() -> int:
    files = sorted(f for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    failures = []
    total = 0
    for fname in files:
        print(f"--- {fname}")
        try:
            mod = runpy.run_path(os.path.join(HERE, fname), run_name="_run_all_")
        except Exception:
            failures.append((fname, "<import>", traceback.format_exc()))
            print(f"FAIL {fname} (import)")
            continue
        for name, fn in sorted(mod.items()):
            if name.startswith("test_") and callable(fn):
                total += 1
                try:
                    fn()
                    print("  PASS", name)
                except Exception:
                    failures.append((fname, name, traceback.format_exc()))
                    print("  FAIL", name)
    print(f"\n{total - len(failures)}/{total} passed")
    for fname, name, tb in failures:
        print(f"\n=== {fname}::{name}\n{tb}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
