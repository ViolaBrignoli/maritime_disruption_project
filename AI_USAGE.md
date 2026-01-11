# AI Usage Documentation

This repository adheres strictly to ethical AI usage policies. AI tools were utilized only in scenarios where their contribution was clear, restricted to correctional suggestions, code optimizations, or guidance. 

## AI Assistance

### **1. `data_prep.py`**
- **Purpose**: Correction of helper functions for data parsing and cleaning. Specifically, `clean_column_names`, `safe_to_numeric`, `clean_dates_to_iso`, and `parse_coordinates_unlocode` were optimized for uniformity and performance.


### **2. `data_loader.py`**
- **Purpose**: Suggestions for robust feature aggregation, handling diverse input schemas, and imputation pipelines. Key focus on `_aggregate_daily_activity_to_yearly` and `_aggregate_daily_chokepoints_to_yearly`.


### **3. `network_graph.py`**
- **Purpose**: Optimization for graph construction with `networkx`, specifically weighted centrality measures like PageRank, closeness, and clustering coefficient. Assisted with efficient edge handling.


### **4. `markov.py`**
- **Purpose**: Guidance on Laplace smoothing and transitioning matrix logic in `train_markov_chain_panel`. Fine-tuned `predict_markov_panel` for operational and pure-forecast modes.


### **5. `models.py`**
- **Purpose**: Helped refine regularization techniques (`apply_identity_dropout`, `add_noise`) and provided architectural clarity for `run_time_series_cv`. 


### **6. `evaluation.py`**
- **Purpose**: Improvements in diagnostic plots (Precision-Recall curves and Lift charts). Suggestions for binning logic in Brier and ECE computation.


### **7. `cascading_predictor.py`**
- **Purpose**: Assistance with shock propagation logic, specifically in `run_cascading_prediction`. Helped fine-tune distance-based decay kernel calculations.


### **8. `stress_tester.py`**
- **Purpose**: Suggestions for axis labeling and annotation handling.


### **9. `fragility_mapper.py`**
- **Purpose**: Improvement in Quadrant Map visualization and aesthetics.


### **10. `investment_optimizer.py`**
- **Purpose**: Suggestions on the stepwise adjustment of investment fractions and auto-selection logic for reductions using elbow-based heuristics.


### **11. `autonomous_agent.py`**
- **Purpose**: Assisted in configuring Monte Carlo trials, CRN shock generation, and surrogate scoring logic. 