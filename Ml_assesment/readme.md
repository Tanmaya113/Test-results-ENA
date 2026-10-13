# Freight Rate Prediction Challenge

Machine learning pipeline and submission artifacts for the freight load rate prediction assessment.

## Project Overview

- **Objective**: Predict posted freight rates (`posted_rate`) across lanes and equipment types while capturing seasonal and day-of-week rate fluctuations.
- **Model Architecture**: Weighted Ensemble of **LightGBM (40%)**, **XGBoost (40%)**, and **ExtraTrees (20%)** with extensive temporal, geographic, and rate-signal interaction features.
- **Validation**: Strict 5-fold `TimeSeriesSplit` cross-validation to respect chronological ordering and avoid lookahead bias.
- **Hold-out Performance (Sep–Oct 2025)**:
  - **R² Score**: 0.9823
  - **MAPE**: 2.97%
  - **MAE**: $54.63
  - **RMSE**: $202.90

---

## Repository Structure

```text
├── train.ipynb                           # End-to-end exploratory analysis, training & evaluation notebook
├── train_model.py                        # Standalone, reproducible training script
├── generate_report.py                    # Script to compile the comprehensive assessment DOCX report
├── score.py                              # Official assessment scoring and chart validation script
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Standard Python/Jupyter gitignore
│
├── Freight_Rate_Prediction_Report.docx   # Detailed technical assessment report
├── scorer_results/
│   └── candidate_december.png            # Generated December 2025 rate prediction chart
│
├── train-test.csv                        # Historical training data (Jan 1 – Oct 31, 2025)
├── validation.csv                        # Target validation data (Nov 1 – Dec 31, 2025)
├── validation-predictions-template.csv   # Template format for validation
├── validation_predictions.csv            # Completed predictions for validation.csv (12,000 loads)
├── december-chart-inputs.csv             # Completed predictions for fixed December lane (31 days)
└── freight-rate-ml-assessment.pdf        # Assessment instructions
```

---

## Quickstart & Reproducibility

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Model Training & Generate Predictions

You can run the interactive notebook or execute the script directly:

```bash
python train_model.py
```

This script:
1. Cleans and engineers 55 temporal, geographic, and market signal features.
2. Performs 5-fold TimeSeries cross-validation.
3. Trains the LightGBM + XGBoost + ExtraTrees ensemble.
4. Generates `validation_predictions.csv` (12,000 loads).
5. Populates `december-chart-inputs.csv` (31 daily rates for Lexington → Fort Wayne).

### 3. Verify with Scorer

```bash
python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv
```

Outputs:
- Validates all 12,000 predictions in `validation_predictions.csv`.
- Validates all 31 days in `december-chart-inputs.csv`.
- Generates `scorer_results/candidate_december.png`.

### 4. Regenerate Submission Report (Optional)

```bash
python generate_report.py
```
Outputs updated `Freight_Rate_Prediction_Report.docx`.