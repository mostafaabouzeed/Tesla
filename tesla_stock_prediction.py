import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler  # Import the StandardScaler for feature scaling

# Importing Dataset
df = pd.read_csv("TESLA.csv", index_col=['Date'], parse_dates=['Date'])
tesla_series = df['Close']

# Descriptive statistics
dataset_stat = df.describe()
close_stat = tesla_series.describe()
median_tesla_close = tesla_series.median()
mean_tesla_close = tesla_series.mean()
std_deviation = tesla_series.std()
print("summary of data set:", dataset_stat)
print("summary of close col:", close_stat)
print("Median of close col:", median_tesla_close)
print("Mean of close col:", mean_tesla_close)
print("Standard Deviation of close col:", std_deviation)

# Train-test split
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Linear Regression
regr = LinearRegression()

# Feature scaling (optional but can improve model performance)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(train[['Open', 'High', 'Low', 'Volume']])
x_test_scaled = scaler.transform(test[['Open', 'High', 'Low', 'Volume']])

y_train = np.asanyarray(train[['Close']])
y_test = np.asanyarray(test[['Close']])

regr.fit(x_train_scaled, y_train)

# Predictions
y_hat = regr.predict(x_test_scaled)

# Evaluation
print("Mean Squared Error (MSE): %.2f" % mean_squared_error(y_test, y_hat))
print('Variance score: %.2f' % regr.score(x_test_scaled, y_test))

# Predict for the next year
future_dates = pd.date_range(start=df.index[-1] + pd.DateOffset(1), periods=365, freq='D')
future_features_scaled = scaler.transform(df[['Open', 'High', 'Low', 'Volume']].values[-1].reshape(1, -1))
future_predictions = []

for _ in range(365):
    prediction = regr.predict(future_features_scaled)
    future_predictions.append(prediction[0])
    future_features_scaled[:, :-1] = np.roll(future_features_scaled[:, :-1], -1, axis=1)
    future_features_scaled[:, -1] = prediction

# Create a DataFrame for the predictions with the corresponding dates
future_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_predictions})
future_df.set_index('Date', inplace=True)

# plotting the close data
plt.figure(figsize=(12, 6))
plt.plot(tesla_series.index, tesla_series, label='Tesla Close Prices ($)', color='green')
plt.title('Tesla Close Prices over the Years ')
plt.xlabel('Date')
plt.ylabel('Close ($)')
plt.legend()
plt.show()
# Creating a 5day Rolling Average
rolling_mean = tesla_series.rolling(window=5).mean()
plt.figure(figsize=(12, 6))
plt.plot(tesla_series.index, tesla_series, label='Tesla', color='green')
plt.plot(rolling_mean.index, rolling_mean, label='Rolling Mean trend', color='orange')
plt.title('Tesla Time Series with 5 Day window Rolling Mean')
plt.xlabel('Date')
plt.ylabel('Exchange Rate')
plt.legend()
plt.show()
# Plotting the predictions
plt.figure(figsize=(12, 6))
plt.plot(df.index, tesla_series, label='Actual Close Prices ($)', color='green')
plt.plot(future_df.index, future_df['Predicted_Close'], label='Predicted Close Prices ($)', color='red', linestyle='dashed')
plt.title('Tesla Close Prices and Predictions')
plt.xlabel('Date')
plt.ylabel('Close ($)')
plt.legend()
plt.show()
# Display the predicted values for the next year
print(future_df)
