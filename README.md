# Salary-Prediction-Project
%%writefile salary_data.csv
Experience,Salary
1,30000
2,35000
3,40000
4,45000
5,50000
6,55000
7,60000
8,65000
9,70000
10,75000

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv("salary_data.csv")

# Split features and target
X = data[['Experience']]
y = data['Salary']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict salary for test data
predictions = model.predict(X_test)

# Plot the results
plt.scatter(X_train, y_train, color="blue", label="Training Data")
plt.scatter(X_test, y_test, color="red", label="Test Data")
plt.plot(X_test, predictions, color="black", linewidth=2, label="Prediction Line")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.show()

# Predict salary for a new experience input
experience = [[7.5]]
predicted_salary = model.predict(experience)
print(f"Predicted Salary for {experience[0][0]} years of experience: {predicted_salary[0]:.2f}")

# Salary Prediction Using Linear Regression

This project predicts salary based on years of experience using a simple Linear Regression model.

## Requirements
- Python
- Pandas
- Matplotlib
- Scikit-learn

## How to Run
1. Install dependencies: `pip install pandas matplotlib scikit-learn`
2. Run the script: `python salary_prediction.py`

## Example Prediction
If experience = 7.5 years, the model predicts a salary of approximately **₹57,500**.
