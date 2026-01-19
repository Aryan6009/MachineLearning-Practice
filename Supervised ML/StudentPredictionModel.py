# import necessary Libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Load the dataset

data =pd.read_csv("student_performance.csv")

# Display the first few rows of the dataset
print(data.head())

# Preprocess the data

print("Checking for missing values:")
print(data.isnull().sum())

#Since there are no missing values, we can proceed further
# Define features and target variable
 
X = data[['weekly_self_study_hours', 'attendance_percentage','class_participation']]
y = data['total_score']



# Create and train the model for predicting total_score (Regression)

model = LinearRegression()
model.fit(X, y)
print("Model trained for predicting total_score.")

print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
# Make predictions on the test set
predicted_total_score = model.predict(X)

# Evaluate the model

mae = mean_absolute_error(y, predicted_total_score)
mse = mean_squared_error(y, predicted_total_score)
r2 = r2_score(y, predicted_total_score)
rmse = np.sqrt(mse)

print("\nModel Evaluation for total_score Prediction:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R2 ): {r2}")

# Visualize the results as a histogram plot(weekly_self_study_hours vs total_score)
plt.figure(figsize=(10,6))
plt.hist(data['total_score'], bins=30, color='blue',  edgecolor='black')
plt.title('Distribution of Total Scores')
plt.xlabel('Total Score')
plt.ylabel('Number of Students')
plt.grid(True)
plt.show()

# visualize the results as a scatter plot (weekly_self_study_hours vs total_score)
plt.figure(figsize=(10,6))
plt.scatter(data['weekly_self_study_hours'], data['total_score'], color='green')
plt.title('Weekly Self Study Hours vs Total Score')
plt.xlabel('Weekly Self Study Hours')
plt.ylabel('Total Score')
plt.grid(True)
plt.show()

#visualize the results as a scatter plot + regression line (weekly_self_study_hours vs total_score)
plt.figure(figsize=(10,6))
plt.scatter(data['weekly_self_study_hours'], data['total_score'], color='green', label='Data Points')
# Regression line
plt.plot(data['weekly_self_study_hours'], predicted_total_score, color='red', linewidth=2, label='Regression Line')
plt.title('Weekly Self Study Hours vs Total Score with Regression Line')
plt.xlabel('Weekly Self Study Hours')
plt.ylabel('Total Score')
plt.legend()
plt.grid(True)
plt.show()

# Predict total_score for a new student
new_student = pd.DataFrame({
    'weekly_self_study_hours': [float(input("Enter weekly self-study hours: "))],
    'attendance_percentage': [float(input("Enter attendance percentage: "))],
    'class_participation': [float(input("Enter class participation score: "))]
})
predicted_score_for_new_student = model.predict(new_student)
print(
    f"Based on {new_student['weekly_self_study_hours'].iloc[0]} hours of weekly self-study, "
    f"{new_student['attendance_percentage'].iloc[0]}% attendance, "
    f"and class participation score of {new_student['class_participation'].iloc[0]}, "
    f"the predicted total score is: {predicted_score_for_new_student[0]:.2f}"
)