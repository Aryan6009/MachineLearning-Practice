import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

data = pd.read_csv("student_score.csv")

X = data[['Hours']]
y = data['Score']

model = LinearRegression()
model.fit(X, y)
predicted_score = model.predict(X)

#evalueate the model
mae = mean_absolute_error(y, predicted_score)
mse = mean_squared_error(y, predicted_score)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

new_hours = float(input("Enter number of hours studied: "))
predicted_new_score = model.predict([[new_hours]])
print(f"Predicted Score for {new_hours} hours of study: {predicted_new_score[0]}")