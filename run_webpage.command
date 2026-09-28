#!/bin/bash
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if ! "$PY" -c "import streamlit,xgboost,Bio" >/dev/null 2>&1; then
  "$PY" -m pip install -r requirements.txt
fi
"$PY" -m streamlit run app.py
