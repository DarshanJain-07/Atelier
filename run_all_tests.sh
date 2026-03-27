#!/bin/bash

set -euo pipefail

export PYTHONPATH=.

if [ -x ".venv/bin/pytest" ]; then
  PYTEST_BIN=".venv/bin/pytest"
elif [ -x ".venv/bin/python" ]; then
  PYTEST_BIN=".venv/bin/python -m pytest"
else
  PYTEST_BIN="python3 -m pytest"
fi

eval "$PYTEST_BIN research_paper_tests -q"
