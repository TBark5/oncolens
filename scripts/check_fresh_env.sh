#!/usr/bin/env bash
# Verify the committed project works from scratch: clone the repo into .fresh_check/,
# create a brand-new virtual environment, install pinned requirements, run the whole
# pipeline from a clean state plus the tests, and confirm the results match the
# committed ones. Everything happens inside the project folder and is removed afterwards.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CHECK="$ROOT/.fresh_check"
PYTHON="${PYTHON:-python}"

rm -rf "$CHECK"
git clone --quiet "$ROOT" "$CHECK/repo"
cd "$CHECK/repo"
"$PYTHON" -m venv .venv
if [ -x .venv/Scripts/python.exe ]; then VPY=.venv/Scripts/python.exe; else VPY=.venv/bin/python; fi
"$VPY" -m pip install --quiet --disable-pip-version-check -r requirements.txt
"$VPY" run_all.py --clean --tests
if git diff --quiet -- results; then
  echo "FRESH ENV CHECK PASSED: results match the committed ones exactly."
else
  echo "FRESH ENV CHECK: results differ from the committed ones:"
  git diff --stat -- results
  exit 1
fi
cd "$ROOT"
rm -rf "$CHECK"
