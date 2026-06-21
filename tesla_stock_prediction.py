import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# ==========================
# LOAD DATA
# ==========================

df = pd.read_csv(
    "TESLA.csv",
    parse_dates=["Date"],
    index_col="Date"
)

print("\nDataset Shape:")
print(df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================
# FEATURE ENGINEERING
# ==========================

df["Lag_1"] = df["Close"].shift(1)
df["Lag_2"] = df["Close"].shift(2)
df["Lag_3"] = df["Close"].shift(3)

df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_20"] = df["Close"].rolling(20).mean()

df["Daily_Return"] = df["Close"].pct_change()

df.dropna(inplace=True)

# ==========================
# EDA
# ==========================

print("\nSummary Statistics")
print(df.describe())

# Close Price Trend

plt.figure(figsize=(12,6))
plt.plot(df.index, df["Close"])
plt.title("Tesla Closing Price")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.grid()
plt.show()

# Moving Averages

plt.figure(figsize=(12,6))
plt.plot(df["Close"], label="Close")
plt.plot(df["MA_5"], label="MA 5")
plt.plot(df["MA_20"], label="MA 20")

plt.title("Moving Averages")
plt.legend()
plt.show()

# ==========================
# CORRELATION
# ==========================

corr = df.corr()

plt.figure(figsize=(10,8))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# ==========================
# MODELING
# ==========================

features = [
    "Lag_1",
    "Lag_2",
    "Lag_3",
    "MA_5",
    "MA_20",
    "Volume"
]

X = df[features]
y = df["Close"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=8,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================

mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Performance")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# ==========================
# ACTUAL VS PREDICTED
# ==========================

plt.figure(figsize=(14,6))

plt.plot(
    y_test.index,
    y_test,
    label="Actual",
    linewidth=2
)

plt.plot(
    y_test.index,
    predictions,
    label="Predicted",
    linestyle="--"
)

plt.title("Actual vs Predicted Tesla Prices")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

# ==========================
# FEATURE IMPORTANCE
# ==========================

importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nFeature Importance")
print(importance)

plt.figure(figsize=(8,5))
plt.barh(
    importance["Feature"],
    importance["Importance"]
)
plt.title("Feature Importance")
plt.show()
