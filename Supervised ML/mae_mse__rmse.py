# MAE(Mean Absolute Error)
"""Calculates the Mean Absolute Error (MAE) between predicted and true values.
MAE is the average of the absolute differences between predicted and actual values."""
#MSE(Mean Squared Error)
"""Calculates the Mean Squared Error (MSE) between predicted and true values.
MSE is the average of the squared differences between predicted and actual values."""
#RMSE(Root Mean Squared Error)
"""Calculates the Root Mean Squared Error (RMSE) between predicted and true values.
RMSE is the square root of the average of the squared differences between predicted and actual values."""

#import necessary libraries
from sklearn.metrics import mean_absolute_error,mean_squared_error
import numpy as np

#real scores

y_true = [90,60,80,100]

#predicted scores

y_pred = [85,70,70,95]

mae = mean_absolute_error(y_true,y_pred)
mse = mean_squared_error(y_true,y_pred)
rmse = np.sqrt(mse)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")