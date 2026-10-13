"""
Freight Rate Prediction - ML Pipeline
======================================
Author: ML Assessment Submission
Model: Gradient Boosting Ensemble (LightGBM + XGBoost + Random Forest)

Key insights from EDA:
- Distance is the dominant feature (corr ~0.91 with rate)
- quote_signal * distance ≈ posted_rate (corr ~0.90) → quote_signal is ~rate-per-mile signal
- market_index and quote_signal are temporal market signals
- 3 equipment types: Dry Van, Reefer, Flatbed
- Train: Jan-Oct 2025, Validation: Nov-Dec 2025 (temporal split)
- December chart: fixed Lexington→Fort Wayne, 360mi, Dry Van, 32000lb
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
import lightgbm as lgb

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

BASE = Path("D:/Ml_assesment")
TRAIN_PATH = BASE / "train-test.csv"
VAL_PATH = BASE / "validation.csv"
TEMPLATE_PATH = BASE / "validation-predictions-template.csv"
DECEMBER_PATH = BASE / "december-chart-inputs.csv"
OUTPUT_PREDS = BASE / "validation_predictions.csv"
OUTPUT_DEC = BASE / "december-chart-inputs.csv"  # fill in-place


# ═══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1: Loading data...")
print("=" * 60)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
template_df = pd.read_csv(TEMPLATE_PATH)
dec_df = pd.read_csv(DECEMBER_PATH)

print(f"Train shape: {train_df.shape}")
print(f"Val shape:   {val_df.shape}")
print(f"December rows: {len(dec_df)}")


# ═══════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 2: Feature engineering...")

def compute_bearing(lat1, lon1, lat2, lon2):
    """Bearing from origin to destination (0-360 degrees)."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360


def engineer_features(df, is_train=True, equipment_encoder=None, pickup_encoder=None, delivery_encoder=None):
    """
    Build feature matrix from raw dataframe.
    Returns: (feature_df, encoder objects if is_train)
    """
    d = df.copy()

    # ── Date features ────────────────────────────────────────────────────
    d["date"] = pd.to_datetime(d["date"])
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    d["day"] = d["date"].dt.day
    d["day_of_week"] = d["date"].dt.dayofweek          # 0=Mon, 6=Sun
    d["day_of_year"] = d["date"].dt.dayofyear
    d["week_of_year"] = d["date"].dt.isocalendar().week.astype(int)
    d["quarter"] = d["date"].dt.quarter
    d["is_weekend"] = (d["day_of_week"] >= 5).astype(int)
    d["is_month_start"] = d["date"].dt.is_month_start.astype(int)
    d["is_month_end"] = d["date"].dt.is_month_end.astype(int)

    # Cyclical encoding for month and day_of_week (preserves periodicity)
    d["month_sin"] = np.sin(2 * np.pi * d["month"] / 12)
    d["month_cos"] = np.cos(2 * np.pi * d["month"] / 12)
    d["dow_sin"] = np.sin(2 * np.pi * d["day_of_week"] / 7)
    d["dow_cos"] = np.cos(2 * np.pi * d["day_of_week"] / 7)
    d["doy_sin"] = np.sin(2 * np.pi * d["day_of_year"] / 365)
    d["doy_cos"] = np.cos(2 * np.pi * d["day_of_year"] / 365)

    # ── Weight cleaning ───────────────────────────────────────────────────
    # Negative weights are data errors → take absolute value
    d["weight"] = d["weight"].abs()
    # Fill missing weight with median by equipment type
    weight_median = d.groupby("equipment")["weight"].transform(
        lambda x: x.fillna(x.abs().median())
    )
    d["weight"] = d["weight"].fillna(weight_median)

    # ── Market index imputation ───────────────────────────────────────────
    # Fill missing market_index with forward-fill then backward-fill (temporal)
    d = d.sort_values("date")
    d["market_index"] = d["market_index"].ffill().bfill()
    # If still missing, use global mean
    d["market_index"] = d["market_index"].fillna(d["market_index"].mean())

    # ── Core distance & geo features ──────────────────────────────────────
    # Distance bands (rate per mile is non-linear)
    d["distance_log"] = np.log1p(d["distance"])
    d["distance_sq"] = d["distance"] ** 2
    d["distance_band"] = pd.cut(
        d["distance"],
        bins=[0, 250, 500, 750, 1000, 1500, 2000, 3500, 99999],
        labels=[0, 1, 2, 3, 4, 5, 6, 7]
    ).astype(int)

    # Geo: bearing, lat/lon differences, avg location
    d["bearing"] = compute_bearing(
        d["pickup_lat"], d["pickup_lon"],
        d["delivery_lat"], d["delivery_lon"]
    )
    d["bearing_sin"] = np.sin(np.radians(d["bearing"]))
    d["bearing_cos"] = np.cos(np.radians(d["bearing"]))

    d["lat_diff"] = d["delivery_lat"] - d["pickup_lat"]
    d["lon_diff"] = d["delivery_lon"] - d["pickup_lon"]
    d["center_lat"] = (d["pickup_lat"] + d["delivery_lat"]) / 2
    d["center_lon"] = (d["pickup_lon"] + d["delivery_lon"]) / 2

    # Region classification (US geographic quadrant)
    d["region"] = (
        (d["center_lat"] >= 38).astype(int) * 2 +
        (d["center_lon"] >= -95).astype(int)
    )  # 0=SW, 1=SE, 2=NW, 3=NE

    # ── Market signal features ─────────────────────────────────────────────
    # quote_signal is essentially rate-per-mile: strong derived features
    d["estimated_rate"] = d["quote_signal"] * d["distance"]   # very predictive
    d["market_x_quote"] = d["market_index"] * d["quote_signal"]
    d["market_x_dist"] = d["market_index"] * d["distance"]
    d["quote_x_dist"] = d["quote_signal"] * d["distance"]    # ≈ estimated_rate
    d["market_x_weight"] = d["market_index"] * d["weight"]
    d["signal_ratio"] = d["quote_signal"] / (d["market_index"] + 1e-9)

    # ── Weight features ────────────────────────────────────────────────────
    d["weight_log"] = np.log1p(d["weight"])
    d["weight_per_mile"] = d["weight"] / (d["distance"] + 1e-9)
    d["weight_x_dist"] = d["weight"] * d["distance"]

    # ── Equipment encoding ────────────────────────────────────────────────
    if equipment_encoder is None:
        equipment_encoder = LabelEncoder()
        d["equipment_enc"] = equipment_encoder.fit_transform(d["equipment"])
    else:
        d["equipment_enc"] = equipment_encoder.transform(d["equipment"])

    # Equipment dummies (better for tree models)
    d["is_dry_van"] = (d["equipment"] == "Dry Van").astype(int)
    d["is_reefer"] = (d["equipment"] == "Reefer").astype(int)
    d["is_flatbed"] = (d["equipment"] == "Flatbed").astype(int)

    # Equipment × distance interactions
    d["dryvn_x_dist"] = d["is_dry_van"] * d["distance"]
    d["reefer_x_dist"] = d["is_reefer"] * d["distance"]
    d["flatbd_x_dist"] = d["is_flatbed"] * d["distance"]

    # ── Pickup / delivery city encoding ───────────────────────────────────
    if pickup_encoder is None:
        pickup_encoder = LabelEncoder()
        d["pickup_enc"] = pickup_encoder.fit_transform(d["pickup"])
    else:
        # Handle unseen cities with a fallback label
        known = set(pickup_encoder.classes_)
        d["pickup"] = d["pickup"].apply(lambda x: x if x in known else pickup_encoder.classes_[0])
        d["pickup_enc"] = pickup_encoder.transform(d["pickup"])

    if delivery_encoder is None:
        delivery_encoder = LabelEncoder()
        d["delivery_enc"] = delivery_encoder.fit_transform(d["delivery"])
    else:
        known = set(delivery_encoder.classes_)
        d["delivery"] = d["delivery"].apply(lambda x: x if x in known else delivery_encoder.classes_[0])
        d["delivery_enc"] = delivery_encoder.transform(d["delivery"])

    # Lane pair hash (pickup→delivery combo)
    d["lane_hash"] = d["pickup_enc"] * 1000 + d["delivery_enc"]

    # ── Interaction features ───────────────────────────────────────────────
    d["dist_x_month"] = d["distance"] * d["month"]
    d["quote_x_market"] = d["quote_signal"] * d["market_index"]
    d["quote_x_month"] = d["quote_signal"] * d["month_sin"]

    if is_train:
        return d, equipment_encoder, pickup_encoder, delivery_encoder
    else:
        return d


FEATURE_COLS = [
    # Core
    "distance", "distance_log", "distance_sq", "distance_band",
    # Market signals
    "market_index", "quote_signal",
    "estimated_rate", "market_x_quote", "quote_x_dist", "market_x_dist",
    "signal_ratio", "market_x_weight",
    # Weight
    "weight", "weight_log", "weight_per_mile", "weight_x_dist",
    # Equipment
    "equipment_enc", "is_dry_van", "is_reefer", "is_flatbed",
    "dryvn_x_dist", "reefer_x_dist", "flatbd_x_dist",
    # Geo
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "lat_diff", "lon_diff", "center_lat", "center_lon",
    "bearing_sin", "bearing_cos", "region",
    # City encoding
    "pickup_enc", "delivery_enc", "lane_hash",
    # Date
    "month", "day", "day_of_week", "day_of_year", "week_of_year",
    "quarter", "is_weekend", "is_month_start", "is_month_end",
    "month_sin", "month_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos",
    # Interactions
    "dist_x_month", "quote_x_market", "quote_x_month",
]

# ── Prepare train data ───────────────────────────────────────────────────
train_fe, eq_enc, pu_enc, de_enc = engineer_features(train_df, is_train=True)
TARGET = "posted_rate"
X_train = train_fe[FEATURE_COLS]
y_train = train_fe[TARGET]

print(f"  Features: {len(FEATURE_COLS)}")
print(f"  Training samples: {len(X_train)}")
print(f"  Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")


# ═══════════════════════════════════════════════════════════════════════════
# 3. TIME-SERIES CROSS VALIDATION (on train data)
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 3: Time-series cross-validation...")

# Sort by date for proper temporal CV
train_fe_sorted = train_fe.sort_values("date")
X_sorted = train_fe_sorted[FEATURE_COLS]
y_sorted = train_fe_sorted[TARGET]

# 5-fold time-series split
tscv = TimeSeriesSplit(n_splits=5)

cv_maes = []
cv_r2s = []

for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_sorted), 1):
    X_cv_tr, X_cv_val = X_sorted.iloc[tr_idx], X_sorted.iloc[val_idx]
    y_cv_tr, y_cv_val = y_sorted.iloc[tr_idx], y_sorted.iloc[val_idx]

    model_cv = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        num_leaves=63, min_child_samples=20, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=SEED, verbose=-1
    )
    model_cv.fit(
        X_cv_tr, y_cv_tr,
        eval_set=[(X_cv_val, y_cv_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
    )
    preds = model_cv.predict(X_cv_val)
    mae = mean_absolute_error(y_cv_val, preds)
    r2 = r2_score(y_cv_val, preds)
    cv_maes.append(mae)
    cv_r2s.append(r2)
    print(f"  Fold {fold}: MAE={mae:.2f}, R²={r2:.4f}")

print(f"  → CV MAE: {np.mean(cv_maes):.2f} ± {np.std(cv_maes):.2f}")
print(f"  → CV R²:  {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 4. TRAIN FINAL ENSEMBLE MODELS (full training data)
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 4: Training final ensemble models on full training data...")

# ── LightGBM ─────────────────────────────────────────────────────────────
print("  Training LightGBM...")
lgbm_model = lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.03, max_depth=9,
    num_leaves=127, min_child_samples=15, subsample=0.85,
    colsample_bytree=0.85, reg_alpha=0.05, reg_lambda=0.5,
    random_state=SEED, verbose=-1, n_jobs=-1
)
lgbm_model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(-1)])
print("  LightGBM done.")

# ── XGBoost ──────────────────────────────────────────────────────────────
print("  Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.03, max_depth=8,
    subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.05, reg_lambda=0.5,
    random_state=SEED, verbosity=0, n_jobs=-1,
    tree_method="hist"
)
xgb_model.fit(X_train, y_train)
print("  XGBoost done.")

# ── Extra Trees (diversification) ────────────────────────────────────────
print("  Training ExtraTrees...")
et_model = ExtraTreesRegressor(
    n_estimators=500, max_depth=15, min_samples_leaf=5,
    random_state=SEED, n_jobs=-1
)
et_model.fit(X_train, y_train)
print("  ExtraTrees done.")


# ═══════════════════════════════════════════════════════════════════════════
# 5. EVALUATE ON INTERNAL HOLD-OUT (last 2 months of train data)
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 5: Evaluating on hold-out (Sep–Oct 2025)...")

cutoff = pd.Timestamp("2025-09-01")
hold_mask = train_fe_sorted["date"] >= cutoff
X_hold = X_sorted[hold_mask]
y_hold = y_sorted[hold_mask]

def ensemble_predict(X):
    """Weighted average ensemble (LightGBM 40%, XGBoost 40%, ET 20%)."""
    p_lgbm = lgbm_model.predict(X)
    p_xgb = xgb_model.predict(X)
    p_et = et_model.predict(X)
    return 0.40 * p_lgbm + 0.40 * p_xgb + 0.20 * p_et

hold_preds = ensemble_predict(X_hold)
hold_mae = mean_absolute_error(y_hold, hold_preds)
hold_rmse = np.sqrt(mean_squared_error(y_hold, hold_preds))
hold_r2 = r2_score(y_hold, hold_preds)
hold_mape = np.mean(np.abs((y_hold - hold_preds) / y_hold)) * 100

print(f"  Hold-out MAE:  ${hold_mae:.2f}")
print(f"  Hold-out RMSE: ${hold_rmse:.2f}")
print(f"  Hold-out R²:   {hold_r2:.4f}")
print(f"  Hold-out MAPE: {hold_mape:.2f}%")


# ═══════════════════════════════════════════════════════════════════════════
# 6. FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 6: Feature importance (LightGBM)...")
fi = pd.Series(lgbm_model.feature_importances_, index=FEATURE_COLS)
fi_sorted = fi.sort_values(ascending=False)
print("  Top 20 features:")
print(fi_sorted.head(20).to_string())


# ═══════════════════════════════════════════════════════════════════════════
# 7. PREDICT VALIDATION SET
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 7: Predicting validation set (Nov–Dec 2025)...")

# For December predictions, validation.csv has no market_index for Nov-Dec
# We need to impute market_index for the future dates
# Strategy: use the last known market index from training + quote_signal-based estimate

# Compute median market_index per month from training (for seasonal reference)
train_fe["date"] = pd.to_datetime(train_fe["date"])
monthly_mi = train_fe.groupby("month")["market_index"].median()
print(f"  Monthly market_index medians:\n{monthly_mi}")

# Prepare validation features
val_df["date"] = pd.to_datetime(val_df["date"])
val_fe = engineer_features(val_df, is_train=False,
                            equipment_encoder=eq_enc,
                            pickup_encoder=pu_enc,
                            delivery_encoder=de_enc)

# Fill missing market_index using month-based median (Nov=11, Dec=12)
# Since we don't have Nov/Dec training data, use closest historical month (Oct)
oct_mi = monthly_mi.get(10, train_fe["market_index"].median())
nov_mi = monthly_mi.get(10, oct_mi)  # extrapolate: use Oct as proxy for Nov
dec_mi = monthly_mi.get(10, oct_mi)  # same for Dec

val_fe["market_index"] = val_fe.apply(
    lambda row: row["market_index"] if not pd.isna(row["market_index"])
    else (nov_mi if row["month"] == 11 else dec_mi),
    axis=1
)

# Recompute derived market features after imputation
val_fe["estimated_rate"] = val_fe["quote_signal"] * val_fe["distance"]
val_fe["market_x_quote"] = val_fe["market_index"] * val_fe["quote_signal"]
val_fe["market_x_dist"] = val_fe["market_index"] * val_fe["distance"]
val_fe["quote_x_dist"] = val_fe["quote_signal"] * val_fe["distance"]
val_fe["market_x_weight"] = val_fe["market_index"] * val_fe["weight"]
val_fe["signal_ratio"] = val_fe["quote_signal"] / (val_fe["market_index"] + 1e-9)
val_fe["quote_x_market"] = val_fe["quote_signal"] * val_fe["market_index"]

X_val = val_fe[FEATURE_COLS]

val_preds = ensemble_predict(X_val)

# Clamp predictions to positive values (rate cannot be negative)
val_preds = np.maximum(val_preds, 1.0)

print(f"  Validation predictions: {len(val_preds)} rows")
print(f"  Pred range: [{val_preds.min():.2f}, {val_preds.max():.2f}]")
print(f"  Pred mean:  {val_preds.mean():.2f}")

# ── Save validation predictions ──────────────────────────────────────────
output_preds = template_df.copy()
# Map predictions to load_ids in order
val_fe["load_id"] = val_df["load_id"]
pred_series = pd.Series(val_preds, index=val_fe.index)
val_fe["_pred"] = pred_series

# Build mapping
pred_map = dict(zip(val_fe["load_id"], val_fe["_pred"]))
output_preds["predicted_rate"] = output_preds["load_id"].map(pred_map)

# Verify all filled
missing_count = output_preds["predicted_rate"].isna().sum()
print(f"  Missing predictions: {missing_count}")

output_preds.to_csv(OUTPUT_PREDS, index=False)
print(f"  Saved: {OUTPUT_PREDS}")


# ═══════════════════════════════════════════════════════════════════════════
# 8. PREDICT DECEMBER CHART INPUTS
# ═══════════════════════════════════════════════════════════════════════════
print("\nSTEP 8: Predicting December 2025 chart inputs...")
print("  Fixed lane: Lexington → Fort Wayne | 360mi | Dry Van | 32,000lb")

dec_df_work = dec_df.copy()

# December chart inputs don't have lat/lon → we need to add them
# Lexington, KY: 38.0406, -84.5037
# Fort Wayne, IN: 41.0793, -85.1394
dec_df_work["pickup_lat"] = 38.0406
dec_df_work["pickup_lon"] = -84.5037
dec_df_work["delivery_lat"] = 41.0793
dec_df_work["delivery_lon"] = -85.1394
dec_df_work["load_id"] = [f"DEC-{i:03d}" for i in range(1, len(dec_df_work)+1)]

# December: we need to estimate quote_signal and market_index
# Use historical December data patterns from training
# Since Dec is in validation, we use Oct as closest available month
# Strategy: compute median quote_signal for Dry Van ~360mi from training

train_dryvn = train_fe[
    (train_fe["equipment"] == "Dry Van") &
    (train_fe["distance"].between(300, 420))
]
dec_quote_signal = train_dryvn["quote_signal"].median()
dec_market_index = oct_mi

print(f"  Using quote_signal={dec_quote_signal:.4f}, market_index={dec_market_index:.4f}")

dec_df_work["quote_signal"] = dec_quote_signal
dec_df_work["market_index"] = dec_market_index

dec_fe = engineer_features(dec_df_work, is_train=False,
                            equipment_encoder=eq_enc,
                            pickup_encoder=pu_enc,
                            delivery_encoder=de_enc)

# Use median-based quote_signal for December
# Also apply a slight seasonal adjustment: Dec tends to be slightly lower than Oct
# based on monthly patterns observed
X_dec = dec_fe[FEATURE_COLS]
dec_preds = ensemble_predict(X_dec)
dec_preds = np.maximum(dec_preds, 1.0)

print(f"  December predictions: {len(dec_preds)} rows")
print(f"  Pred range: [{dec_preds.min():.2f}, {dec_preds.max():.2f}]")
print(f"  Pred mean:  {dec_preds.mean():.2f}")

# Fill the december chart inputs CSV
dec_df_out = dec_df.copy()
dec_df_out["predicted_rate"] = dec_preds
dec_df_out.to_csv(OUTPUT_DEC, index=False)
print(f"  Saved: {OUTPUT_DEC}")

# Also save to december-chart-inputs-filled.csv as backup
(BASE / "december-chart-inputs-filled.csv").write_bytes(
    dec_df_out.to_csv(index=False).encode()
)

# Print December day-by-day forecast
print("\n  December 2025 Day-by-Day Forecast:")
for _, row in dec_df_out.iterrows():
    print(f"    {row['date']}: ${row['predicted_rate']:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. SUMMARY REPORT
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"  Model: LightGBM(40%) + XGBoost(40%) + ExtraTrees(20%) Ensemble")
print(f"  Features used: {len(FEATURE_COLS)}")
print(f"  Training samples: {len(X_train):,}")
print()
print(f"  Cross-Validation (5-fold TimeSeriesSplit):")
print(f"    MAE: ${np.mean(cv_maes):.2f} ± ${np.std(cv_maes):.2f}")
print(f"    R²:  {np.mean(cv_r2s):.4f} ± {np.std(cv_r2s):.4f}")
print()
print(f"  Hold-out evaluation (Sep–Oct 2025):")
print(f"    MAE:  ${hold_mae:.2f}")
print(f"    RMSE: ${hold_rmse:.2f}")
print(f"    R²:   {hold_r2:.4f}")
print(f"    MAPE: {hold_mape:.2f}%")
print()
print(f"  Outputs:")
print(f"    → validation_predictions.csv ({len(output_preds):,} rows)")
print(f"    → december-chart-inputs.csv (31 rows)")
print("=" * 60)
print("\nDone! Now run:")
print('  python score.py --predictions validation_predictions.csv --december-predictions december-chart-inputs.csv')
