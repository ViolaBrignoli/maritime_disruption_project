# MARITIME DISRUPTION PROJECT - Viola Brignoli 25444845

Author
- Viola Brignoli — Data Science and Advanced Programming 2025

## Research question

How do localized disruptions at major global ports propagate through the maritime port network, and which targeted resilience investments or mitigations most effectively reduce network-wide impacts?

## Overview

The pipeline integrates network science (country/port centrality and community structure), port-level panel features, and supervised machine learning (Random Forest, XGBoost, KNN, MLP, Logistic Regression, ExtraTrees) with a Markov baseline to deliver:
- predictive models for cascade risk,
- batch stress-test simulations for scenarios,
- structural fragility diagnostics,
- an Economic Value-of-Resilience (EVR) optimizer,
- and an Autonomous Resilience Agent (ARA) that recommends sequential investments.

## Setup

# Environment

This project uses a local Python virtual environment and a requirements.txt file. 

macOS / Linux (bash / zsh):
1. Create a local venv:
python3 -m venv .venv

2. Activate it:
source .venv/bin/activate

3. Install requirements:
pip install -r requirements.txt

Windows (PowerShell):
1. Create a local venv:
python -m venv .venv

2. Activate it (PowerShell):
.\.venv\Scripts\Activate.ps1

3. Install requirements:
pip install -r requirements.txt

## Usage
After inseriting files in `data/raw/` (as instructed below). 
Run in activated environment (as instructed previously) -> python main.py

## Raw files layout

The pipeline expects the raw input files to be placed inside `data/raw/` in a small set of
subfolders. I will provide the raw files in a folder already sorted separately (via Google Drive Link).

Raw folder layout:
- data/raw/
  - container_port_traffic_worldbank/
    - API_IS.SHP.GOOD.TU_...csv
    - Metadata_Country_API_...csv
  - global_maritime_pirate_attacks/
    - country_codes.csv
    - country_indicators.csv
    - pirate_attacks.csv
    - pirate_attacks.csv.zip
  - imf_portwatch/
    - chokepoints_imf/
        - Chokepoints.csv
    - daily_chokepoints_transitcalls_tradevolumes/
        - Daily_Chokepoint_Transit_Calls_and_Trade_Volume_Estimates (1).csv
    - daily_port_activity_imf/
        - Daily_Port_Activity_Data_and_Trade_Estimates (2).csv
    - disruptions_imf/
        - portwatch_disruptions_database_-3602226124776604501.csv
    - disruptions_with_ports/
        - disruptions_with_ports.csv
    - ports_imf/
        - ports_imf.csv
  - unctad/
    - combined_linear_shipping_bilateral_connectivity_index/... (several)
    - container_port_throughput/... (several)
    - linear_shipping_bilateral_connectivity_index/... (several)
    - lsci/... (several)
    - plsci/... (several)
    - seaborne_trade_cargotype/... (several)
    - trade-and-transport/... (several)
  - unlocode_ports/
    - unlocode_ports.csv

Important: the ETL script (`src/data_prep.py`) contains logic to try multiple candidate filenames
for each logical dataset. This is intentional, as some raw file numbers changed during project development(after regenerating source files). If raw files follow the folders above, the ETL will standardize
them automatically and save canonical CSVs into `data/processed/`. As mentioned earlier, the files will already be separated accordingly in the Google Drive folder provided via the shared link in the email.

## Project structure (visual)

maritime_disruption_project/
├── data/
│   ├── raw/                # raw input folders (place the provided raw archives here)
│   └── processed/          # created automatically by main.py / ETL
├── results/                # artifacts produced by main.py (plots, CSVs, models)
├── src/                    # project source modules
│   ├── autonomous_agent.py  # ARA implementation
│   ├── cascading_predictor.py # cascade simulation and stress-test orchestrator
│   ├── data_loader.py      # port-year panel loader
│   ├── data_prep.py        # data preparation pipeline
│   ├── evaluation.py       # models evaluation module
│   ├── fragility_mapper.py # quadrant fragility map
│   ├── investment_optimizer.py # investment optimization EVR
│   ├── markov.py           # base line 
│   ├── models.py           # multi model pipeline ML
│   ├── network_graph.py    # network graph construction and feature engineering
│   └── stress_tester.py    # cumulative risk plot
├── AI_USAGE.md
├── main.py                 # orchestration / reproducible entry point
├── project_report.md
├── PROPOSAL.md
├── README.md
└── requirements.txt

## Results

Environment & panel
- Ports in panel: 1,978
- Panel rows: 70,821 (years 1993–2025)
- Train rows (≤2020): 58,714
- Test rows (≥2021): 12,107
- Feature count (encoded): 487

Markov baseline (panel-aware)
- Transition matrix:
  - P(next=1 | current=0) = 0.0017
  - P(next=1 | current=1) = 0.7605
- Test samples: 12,107
- Accuracy: 0.9377
- F1: 0.0575

Model training & comparison (test ROC AUC shown)
- ExtraTrees (et): Test ROC AUC = 0.9158
- XGBoost (xgb): Test ROC AUC = 0.9029
- Random Forest (rf): Test ROC AUC = 0.9127
- MLP (mlp): Test ROC AUC = 0.9333
- Logistic Regression (lr): Test ROC AUC = 0.9357
- KNN (knn): Test ROC AUC = 0.9409 ← Selected as best model

Mutual-information Top-10 features (MI snapshot)
1. events__n_events (0.0875)
2. meta__lat (0.0407)
3. meta__lon (0.0310)
4. network__network__net_pagerank (0.0303)
5. activity__import_volume_total_tonnes__sum (0.0275)
6. events__n_events_pe (0.0267)
7. network__network__net_closeness (0.0262)
8. network__network__net_clustering (0.0247)
9. activity__import_volume_total_tonnes__mean (0.0239)
10. country__lsci_index (0.0236)

Evaluation artifacts produced
- `results/final_model_comparison.csv` (per-model metrics)
- ROC/PR/Calibration/Lift plots (PNG) per model
- `results/best_model_final.joblib` (persisted best model)
- `results/best_model_metadata.json`

Batch stress-test (scenario validation)
- Scenarios run: 6 (Singapore, Suez, Shanghai, Rotterdam, Durban, Valparaiso)
- Affected ports reported: 52 total
- Confidence screening (XGBoost):
  - High: 7
  - Moderate: 11
  - Low: 25
- Example top affected ports (sorted by Risk_Increase): tuticorin (Shanghai scenario), haimen, veracruz, hofu, guangzhou, antwerp, ...  
- Cumulative risk plot saved: `results/multi_scenario_risk_summary.png`

Fragility mapping
- Structural fragility map saved: `results/structural_fragility_map.png`
- Scenario Intensity (Ip) ranking (example):
  1. Shanghai — Ip = 0.2540
  2. Durban — Ip = 0.0540
  3. Suez — Ip = 0.0380
  4. Singapore — Ip = 0.0240
  5. Rotterdam — Ip = 0.0200
  6. Valparaiso — Ip = 0.0080

EVR Investment Optimization (example outputs)
- Investment recommendations persisted: `results/investment_recommendations.csv`
- Pareto frontier persisted: `results/investment_pareto_frontier.csv`
- Investment ranking plot saved: `results/resilience_investment_ranking.png`

Top EVR highlights (marginal risk reduction per investment unit)
- Shanghai — Investment ≈ 2.231 → Selected reduction ≈ 26.78% → Global risk reduction = 0.1120 → ROI ≈ 0.0502
- Singapore — Investment ≈ 1.250 → Selected reduction ≈ 21.41% → Global risk reduction = 0.0060 → ROI ≈ 0.0048
- Durban, Valparaiso, Rotterdam, Suez — follow in the saved CSV

Autonomous Resilience Agent (ARA) — policy sequence
- Baseline systemic risk (mean ± std): 6.820000 ± 0.022000
- Final recommended sequence (example):
  1. Invest in Shanghai — expected marginal Δrisk = 4.841000; residual system risk ≈ 1.979000
  2. Invest in Suez — expected marginal Δrisk = 0.122000; residual system risk ≈ 1.857000
- ARA sequence persisted to: `results/ara_policy_sequence.csv` and `results/ara_policy_summary.json`

All artifacts from this run are in `results/`:
- Models: `best_model_final.joblib`, `xgb_model_final.joblib`
- Tables: `final_model_comparison.csv`, `top_features_mi_for_cascade_investigation.csv`, `investment_recommendations.csv`, `investment_pareto_frontier.csv`, `ara_policy_sequence.csv`
- Plots: `multi_scenario_risk_summary.png`, `structural_fragility_map.png`, `resilience_investment_ranking.png`, evaluation PNGs

## Requirements

- Python 3.11  
- Core packages: numpy, pandas, scikit-learn, joblib, xgboost, matplotlib, seaborn, networkx, python-louvain
- See requirements.txt