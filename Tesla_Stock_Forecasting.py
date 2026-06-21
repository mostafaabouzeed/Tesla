# ============================================================
# TESLA STOCK FORECASTING PORTFOLIO PROJECT
# Mostafa Abouzeed
# ============================================================

import os
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

# ============================================================
# CREATE IMAGE FOLDER
# ============================================================

os.makedirs("images", exist_ok=True)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "TESLA.csv",
    parse_dates=["Date"]
)

df.sort_values("Date", inplace=True)
df.set_index("Date", inplace=True)

print("=" * 60)
print("TESLA STOCK FORECASTING PROJECT")
print("=" * 60)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ============================================================
# FEATURE ENGINEERING
# ============================================================

df["Lag_1"] = df["Close"].shift(1)
df["Lag_2"] = df["Close"].shift(2)
df["Lag_3"] = df["Close"].shift(3)

df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_10"] = df["Close"].rolling(10).mean()
df["MA_20"] = df["Close"].rolling(20).mean()

df["Return"] = df["Close"].pct_change()

df["Volatility"] = (
    df["High"] - df["Low"]
)

df.dropna(inplace=True)

# ============================================================
# EDA
# ============================================================

print("\nSummary Statistics")
print(df.describe())

# ============================================================
# PRICE TREND
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(
    df.index,
    df["Close"],
    linewidth=2
)

plt.title("Tesla Closing Price History")

plt.xlabel("Date")
plt.ylabel("Price ($)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/01_price_history.png",
    dpi=300
)

plt.show()

# ============================================================
# MOVING AVERAGES
# ============================================================

plt.figure(figsize=(12,6))

plt.plot(
    df["Close"],
    label="Close"
)

plt.plot(
    df["MA_5"],
    label="MA 5"
)

plt.plot(
    df["MA_20"],
    label="MA 20"
)

plt.title("Moving Averages")

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/02_moving_averages.png",
    dpi=300
)

plt.show()

# ============================================================
# DISTRIBUTION
# ============================================================

plt.figure(figsize=(10,6))

plt.hist(
    df["Close"],
    bins=30
)

plt.title(
    "Distribution of Tesla Prices"
)

plt.xlabel("Price")

plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "images/03_distribution.png",
    dpi=300
)

plt.show()

# ============================================================
# CORRELATION MATRIX
# ============================================================

corr = df.corr()

plt.figure(figsize=(10,8))

plt.imshow(
    corr,
    cmap="coolwarm",
    aspect="auto"
)

plt.colorbar()

plt.xticks(
    range(len(corr.columns)),
    corr.columns,
    rotation=90
)

plt.yticks(
    range(len(corr.columns)),
    corr.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
    "images/04_correlation_matrix.png",
    dpi=300
)

plt.show()

# ============================================================
# FEATURES
# ============================================================

features = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "MA_5",
    "MA_10",
    "MA_20",
    "Volume",
    "Volatility"
]

X = df[features]

y = df["Close"]

# ============================================================
# TIME SERIES SPLIT
# ============================================================

split = int(len(df) * 0.80)

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = y.iloc[:split]
y_test = y.iloc[split:]

# ============================================================
# MODEL COMPARISON
# ============================================================

models = {
    "Linear Regression":
        LinearRegression(),

    "Random Forest":
        RandomForestRegressor(
            n_estimators=300,
            random_state=42
        ),

    "Gradient Boosting":
        GradientBoostingRegressor(
            random_state=42
        )
}

results = []

best_model = None
best_predictions = None
best_r2 = -999

for name, model in models.items():

    model.fit(
        X_train,
        y_train
    )

    pred = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            pred
        )
    )

    r2 = r2_score(
        y_test,
        pred
    )

    results.append([
        name,
        mae,
        rmse,
        r2
    ])

    if r2 > best_r2:

        best_r2 = r2
        best_model = model
        best_predictions = pred

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "MAE",
        "RMSE",
        "R2"
    ]
)

print("\nModel Results")
print(results_df)

results_df.to_csv(
    "model_comparison.csv",
    index=False
)

# ============================================================
# MODEL COMPARISON CHART
# ============================================================

plt.figure(figsize=(8,5))

plt.bar(
    results_df["Model"],
    results_df["R2"]
)

plt.title(
    "Model Comparison (R²)"
)

plt.ylabel("R² Score")

plt.tight_layout()

plt.savefig(
    "images/05_model_comparison.png",
    dpi=300
)

plt.show()

# ============================================================
# ACTUAL VS PREDICTED
# ============================================================

plt.figure(figsize=(14,6))

plt.plot(
    y_test.index,
    y_test,
    label="Actual"
)

plt.plot(
    y_test.index,
    best_predictions,
    linestyle="--",
    label="Predicted"
)

plt.title(
    "Actual vs Predicted Tesla Prices"
)

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/06_actual_vs_predicted.png",
    dpi=300
)

plt.show()

# ============================================================
# FEATURE IMPORTANCE
# ============================================================

if hasattr(best_model, "feature_importances_"):

    importance = pd.DataFrame({

        "Feature": features,

        "Importance":
        best_model.feature_importances_

    })

    importance.sort_values(
        by="Importance",
        ascending=False,
        inplace=True
    )

    plt.figure(figsize=(8,5))

    plt.barh(
        importance["Feature"],
        importance["Importance"]
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        "images/07_feature_importance.png",
        dpi=300
    )

    plt.show()

# ============================================================
# 30 DAY FORECAST
# ============================================================

future_days = 30

future_predictions = []

last_row = X.iloc[-1].copy()

for _ in range(future_days):

    prediction = best_model.predict(
        pd.DataFrame([last_row])
    )[0]

    future_predictions.append(
        prediction
    )

    last_row["Lag_3"] = last_row["Lag_2"]
    last_row["Lag_2"] = last_row["Lag_1"]
    last_row["Lag_1"] = prediction

future_dates = pd.date_range(
    start=df.index[-1] + pd.Timedelta(days=1),
    periods=future_days
)

forecast_df = pd.DataFrame({

    "Date":
    future_dates,

    "Forecast_Close":
    future_predictions

})

forecast_df.to_csv(
    "future_predictions.csv",
    index=False
)

# ============================================================
# FUTURE FORECAST CHART
# ============================================================

plt.figure(figsize=(14,6))

plt.plot(
    df.index,
    df["Close"],
    label="Historical"
)

plt.plot(
    future_dates,
    future_predictions,
    color="red",
    linestyle="--",
    linewidth=3,
    label="30 Day Forecast"
)

plt.title(
    "Tesla 30-Day Forecast"
)

plt.xlabel("Date")
plt.ylabel("Price")

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "images/08_future_forecast.png",
    dpi=300
)

plt.show()

print("\nProject Finished Successfully")
print("Images saved in images/ folder")
