













































































import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data: Number of hours spent driving (X) and Risk Score (y)
X = np.array([10, 9, 2, 15, 10, 16, 11, 16]).reshape(-1, 1)
y = np.array([95, 80, 10, 50, 45, 98, 38, 93])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict risk score for 20 hours of driving
hours_spent = np.array([[20]])
predicted_risk_score = model.predict(hours_spent)

# Display the result
print(f"Predicted risk score for 20 hours of driving: {predicted_risk_score[0]:.2f}")

# Plot the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Best fit line')
plt.scatter(hours_spent, predicted_risk_score, color='green', marker='x', s=100, label='Prediction for 20 hours')
plt.xlabel('Number of hours spent driving')
plt.ylabel('Risk Score')
plt.legend()
plt.show()
