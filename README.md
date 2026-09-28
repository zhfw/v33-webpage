# CFPS Manufacturability Predictor v3.3 — first web prototype

This is the first runnable web page built from the uploaded public training datasets and the current trained CPU models.

## Current model-backed outputs
- General solubility probability: 5-fold XGBoost ensemble.
- CFPS/eSOL soluble fraction: 5-fold quantitative eSOL ensemble (21600g / 30 min).
- Operational aggregation/insoluble fraction = 1 - predicted soluble fraction.
- RP3 recombinant E. coli production probability, protein-only.
- RP3 protein+DNA production probability when coding DNA is supplied.
- NESG expression and solubility ordinal scores (0–5).
- Sequence-level diagnostic risks and redesign suggestions.

## Not yet model-backed
- Absolute CFPS yield in mg/mL.
- CHO secretion/yield.
- RP3Net 256-D STP inference inside the web request path (planned upgrade after CPU STP cache is complete).

## Run on macOS/Linux
Double-click `run_webpage.command` on macOS, or run:

```bash
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

The browser should open to the local Streamlit URL.

## Federated-ready architecture
The Federated Network tab shows the validated three-party CPU baseline. Later the same UI can switch from 42 engineered features to 256-D RP3Net-STP + engineered features without changing the user-facing input/output design.
