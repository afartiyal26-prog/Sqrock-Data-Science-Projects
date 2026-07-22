import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

# Load dataset
data = pd.read_csv("dataset/sales_data.csv")

# Show first 5 rows
print("First 5 Rows:")
print(data.head())

# Convert Date column to datetime
data["Date"] = pd.to_datetime(data["Date"])

# Create Month feature
data["Month"] = data["Date"].dt.month

print("\nDataset Information:")
print(data.info())




# Professional Line Chart
plt.figure(figsize=(12,6))

plt.plot(
    data["Date"],
    data["Sales"],
    color="#2563EB",
    linewidth=3,
    marker="o",
    markersize=8,
    markerfacecolor="white",
    markeredgewidth=2
)

plt.title("Monthly Sales Trend Analysis", fontsize=18, fontweight="bold")
plt.xlabel("Date", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# Professional Bar Chart
plt.figure(figsize=(12,6))

bars = plt.bar(
    data["Month"],
    data["Sales"],
    color="#14B8A6",
    edgecolor="black",
    linewidth=1.2
)

# Value labels
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        height + 2,
        f"{int(height)}",
        ha="center",
        fontsize=10,
        fontweight="bold"
    )

plt.title("Monthly Sales Comparison", fontsize=18, fontweight="bold")
plt.xlabel("Month", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.xticks(data["Month"])
plt.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()




# -------------------------------
# Machine Learning - Linear Regression
# -------------------------------

# Features and Target
X = data[["Month"]]
y = data["Sales"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("\nModel Performance")
print("-------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# Predict Future Sales (Month 13)
future_month = [[13]]
future_sales = model.predict(future_month)

print(f"\nPredicted Sales for Month 13: {future_sales[0]:.2f}")

# -------------------------------
# Actual vs Predicted Graph
# -------------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    X_test,
    y_test,
    color="#2563EB",
    s=120,
    label="Actual Sales"
)

plt.plot(
    X_test,
    y_pred,
    color="#DC2626",
    linewidth=3,
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales", fontsize=18, fontweight="bold")
plt.xlabel("Month", fontsize=12)
plt.ylabel("Sales", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()



# -------------------------------
# Pie Chart
# -------------------------------
plt.figure(figsize=(8,8))

plt.pie(
    data["Sales"],
    labels=data["Month"],
    autopct="%1.1f%%",
    startangle=90,
    shadow=True
)

plt.title("Monthly Sales Distribution", fontsize=18, fontweight="bold")
plt.show()


# -------------------------------
# Histogram
# -------------------------------
plt.figure(figsize=(10,6))

plt.hist(
    data["Sales"],
    bins=6,
    edgecolor="black"
)

plt.title("Sales Distribution", fontsize=18, fontweight="bold")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()


# -------------------------------
# Box Plot
# -------------------------------
plt.figure(figsize=(6,6))

plt.boxplot(
    data["Sales"],
    patch_artist=True
)

plt.title("Sales Outlier Detection", fontsize=18, fontweight="bold")
plt.ylabel("Sales")

plt.show()


# -------------------------------
# Moving Average Trend
# -------------------------------
data["Moving_Average"] = data["Sales"].rolling(window=3).mean()

plt.figure(figsize=(12,6))

plt.plot(
    data["Date"],
    data["Sales"],
    label="Sales",
    linewidth=3
)

plt.plot(
    data["Date"],
    data["Moving_Average"],
    linewidth=3,
    linestyle="--",
    label="3-Month Moving Average"
)

plt.title("Sales vs Moving Average", fontsize=18, fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()



# -------------------------------
# Prediction Table
# -------------------------------

results = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": y_pred.round(2)
})

print("\nPrediction Results")
print(results)


# -------------------------------
# Future Sales Forecast (Next 6 Months)
# -------------------------------

future_months = pd.DataFrame({
    "Month": [13, 14, 15, 16, 17, 18]
})

future_predictions = model.predict(future_months)

forecast = pd.DataFrame({
    "Month": future_months["Month"],
    "Predicted Sales": future_predictions.round(2)
})

print("\nFuture Sales Forecast")
print(forecast)


# -------------------------------
# Future Forecast Graph
# -------------------------------

plt.figure(figsize=(12,6))

plt.plot(
    forecast["Month"],
    forecast["Predicted Sales"],
    marker="o",
    linewidth=3
)

for x, y in zip(forecast["Month"], forecast["Predicted Sales"]):
    plt.text(x, y + 2, f"{y:.0f}", ha="center", fontsize=10)

plt.title("Future Sales Forecast", fontsize=18, fontweight="bold")
plt.xlabel("Future Month")
plt.ylabel("Predicted Sales")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()