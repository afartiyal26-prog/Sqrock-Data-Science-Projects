import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load Dataset
data = pd.read_csv("dataset/student_data.csv")

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
# Graph 1 - Final Score Distribution
# -------------------------------

plt.figure(figsize=(10,6))

plt.hist(
    data["Final_Score"],
    bins=6,
    edgecolor="black"
)

plt.title("Final Score Distribution", fontsize=18, fontweight="bold")
plt.xlabel("Final Score")
plt.ylabel("Number of Students")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# -------------------------------
# Graph 2 - Hours Studied vs Final Score
# -------------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    data["Hours_Studied"],
    data["Final_Score"],
    s=120
)

plt.title("Hours Studied vs Final Score", fontsize=18, fontweight="bold")
plt.xlabel("Hours Studied")
plt.ylabel("Final Score")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# -------------------------------
# Graph 3 - Attendance vs Final Score
# -------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    data["Attendance"],
    data["Final_Score"],
    marker="o",
    linewidth=3
)

plt.title("Attendance vs Final Score", fontsize=18, fontweight="bold")
plt.xlabel("Attendance (%)")
plt.ylabel("Final Score")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()


# -------------------------------
# Machine Learning - Linear Regression
# -------------------------------

# Features and Target
X = data[["Hours_Studied", "Attendance", "Previous_Score"]]
y = data["Final_Score"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = LinearRegression()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

print("\nModel Performance")
print("---------------------------")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")

# Sample Prediction
new_student = pd.DataFrame({
    "Hours_Studied": [8],
    "Attendance": [90],
    "Previous_Score": [82]
})

predicted_score = model.predict(new_student)

print("\nPredicted Final Score:")
print(f"{predicted_score[0]:.2f}")



# -------------------------------
# Actual vs Predicted Graph
# -------------------------------

plt.figure(figsize=(10,6))

plt.scatter(
    y_test,
    y_pred,
    s=120,
    label="Predictions"
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linewidth=3,
    label="Ideal Prediction"
)

plt.title("Actual vs Predicted Scores", fontsize=18, fontweight="bold")
plt.xlabel("Actual Score")
plt.ylabel("Predicted Score")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

plt.tight_layout()
plt.show()



# -------------------------------
# Prediction Results Table
# -------------------------------

results = pd.DataFrame({
    "Actual Score": y_test.values,
    "Predicted Score": y_pred.round(2)
})

print("\nPrediction Results")
print(results)


# -------------------------------
# Future Student Prediction
# -------------------------------

future_students = pd.DataFrame({
    "Hours_Studied": [6, 7, 8, 9, 10],
    "Attendance": [82, 85, 90, 94, 97],
    "Previous_Score": [70, 75, 80, 85, 90]
})

future_predictions = model.predict(future_students)

forecast = future_students.copy()
forecast["Predicted Final Score"] = future_predictions.round(2)

print("\nFuture Student Predictions")
print(forecast)


# -------------------------------
# Future Prediction Graph
# -------------------------------

plt.figure(figsize=(10,6))

plt.plot(
    range(1, len(forecast)+1),
    forecast["Predicted Final Score"],
    marker="o",
    linewidth=3
)

plt.title("Future Student Score Prediction", fontsize=18, fontweight="bold")
plt.xlabel("Student Number")
plt.ylabel("Predicted Final Score")
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()