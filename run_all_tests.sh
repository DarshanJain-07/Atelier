#!/bin/bash

set -euo pipefail

export PYTHONPATH=.

EVOLUTION_MODE=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --evolution)
      if [ "$#" -lt 2 ]; then
        echo "Missing value for --evolution. Use: with, without, or both." >&2
        exit 1
      fi
      EVOLUTION_MODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      echo "Usage: $0 [--evolution with|without|both]" >&2
      exit 1
      ;;
  esac
done

if [ -n "$EVOLUTION_MODE" ]; then
  case "$EVOLUTION_MODE" in
    with|without|both|on|off|true|false)
      export RESEARCH_TEST_EVOLUTION_MATRIX="$EVOLUTION_MODE"
      ;;
    *)
      echo "Invalid evolution mode: $EVOLUTION_MODE" >&2
      echo "Use one of: with, without, both" >&2
      exit 1
      ;;
  esac
fi

if [ -x ".venv/bin/pytest" ]; then
  PYTEST_BIN=".venv/bin/pytest"
elif [ -x ".venv/bin/python" ]; then
  PYTEST_BIN=".venv/bin/python -m pytest"
else
  PYTEST_BIN="python3 -m pytest"
fi

eval "$PYTEST_BIN research_paper_tests -q"
