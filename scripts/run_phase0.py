#!/usr/bin/env python
"""Run all Phase 0 tests and report results."""
import subprocess
import sys

tests = [
    ("Environment validation", "scripts/check_environment.py"),
    ("Whisper encoder",        "tests/smoke_test_whisper.py"),
    ("Quantum kernel",         "tests/smoke_test_quantum_kernel.py"),
    ("Noise injection",        "tests/smoke_test_noise.py"),
]

results = {}
for name, script in tests:
    print(f"\n{'='*60}")
    print(f"RUNNING: {name}")
    print(f"{'='*60}")
    r = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        cwd=".",
    )
    results[name] = r.returncode == 0

print()
print("=" * 60)
print("PHASE 0 STATUS")
print("-" * 40)
for name, passed in results.items():
    status = "PASS" if passed else "FAIL"
    print(f"  {name:30s} {status}")
overall = all(results.values())
print(f"  {'Overall':30s} {'PASS' if overall else 'FAIL'}")
print("=" * 60)

sys.exit(0 if overall else 1)
