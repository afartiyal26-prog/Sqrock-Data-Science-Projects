import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Load Dataset
data = pd.read_csv("dataset/churn_data.csv")

# Display First 5 Rows
print("First 5 Rows:")
print(data.head())

# Dataset Information
print("\nDataset Information:")
print(data.info())

# Missing Values
print("\nMissing Values:")
print(data.isnull().sum())

# Statistical Summary
print("\nStatistical Summary:")
print(data.describe())



# -------------------------------
# Graph 1: Churn Distribution
# -------------------------------

plt.figure(figsize=(8,6))
data["Churn"].value_counts().plot(kind="bar")
plt.title("Customer Churn Distribution", fontsize=18, fontweight="bold")
plt.xlabel("Churn (0 = No, 1 = Yes)")
plt.ylabel("Number of Customers")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# -------------------------------
# Graph 2: Monthly Bill Distribution
# -------------------------------

plt.figure(figsize=(8,6))
plt.hist(data["Monthly_Bill"], bins=6, edgecolor="black")
plt.title("Monthly Bill Distribution", fontsize=18, fontweight="bold")
plt.xlabel("Monthly Bill")
plt.ylabel("Customers")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# -------------------------------
# Graph 3: Age vs Monthly Bill
# -------------------------------

plt.figure(figsize=(8,6))
plt.scatter(data["Age"], data["Monthly_Bill"], s=120)
plt.title("Age vs Monthly Bill", fontsize=18, fontweight="bold")
plt.xlabel("Age")
plt.ylabel("Monthly Bill")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()



# -------------------------------
# Machine Learning - Logistic Regression
# -------------------------------

# Features and Target
X = data[["Age", "Monthly_Bill", "Contract_Months"]]
y = data["Churn"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy")
print("-----------------------")
print(f"Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix")
print(cm)

# New Customer Prediction
new_customer = pd.DataFrame({
    "Age": [32],
    "Monthly_Bill": [850],
    "Contract_Months": [12]
})

prediction = model.predict(new_customer)

print("\nNew Customer Prediction")
if prediction[0] == 1:
    print("Customer is likely to Churn.")
else:
    print("Customer is likely to Stay.")



    # -------------------------------
# Actual vs Predicted Churn
# -------------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    range(len(y_test)),
    y_test,
    s=120,
    label="Actual"
)

plt.scatter(
    range(len(y_pred)),
    y_pred,
    s=120,
    label="Predicted"
)

plt.title("Actual vs Predicted Churn", fontsize=18, fontweight="bold")
plt.xlabel("Customer")
plt.ylabel("Churn (0 = No, 1 = Yes)")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()


# -------------------------------
# Prediction Results Table
# -------------------------------

results = pd.DataFrame({
    "Actual Churn": y_test.values,
    "Predicted Churn": y_pred
})

print("\nPrediction Results")
print(results)


# -------------------------------
# Future Customer Prediction
# -------------------------------

future_customers = pd.DataFrame({
    "Age": [25, 30, 35, 40, 45],
    "Monthly_Bill": [600, 800, 950, 1100, 1250],
    "Contract_Months": [6, 12, 18, 24, 36]
})

future_predictions = model.predict(future_customers)

forecast = future_customers.copy()
forecast["Predicted Churn"] = future_predictions

print("\nFuture Customer Predictions")
print(forecast)


# -------------------------------
# Future Customer Prediction Graph
# -------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    range(1, len(forecast)+1),
    forecast["Predicted Churn"],
    marker="o",
    linewidth=3
)

plt.title("Future Customer Churn Prediction", fontsize=18, fontweight="bold")
plt.xlabel("Customer Number")
plt.ylabel("Predicted Churn (0 = Stay, 1 = Churn)")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()