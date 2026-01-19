from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split


# scaler = StandardScaler()
# X_scaled = scaler.fit_transform()

# scaler = MinMaxScaler()
# X_minmax = scaler.fit_transform()

data = {
    'study_hours': [2, 4, 6, 8 , 10],
    'scores': [20, 40, 60, 80, 100]
}

df = pd.DataFrame(data)

#standard scaling
standard_scaler = StandardScaler()
standard_scaled = standard_scaler.fit_transform(df)

print("Standard Scaled Data:")
print(pd.DataFrame(standard_scaled, columns=df.columns))

#min-max scaling

minmax_scaler = MinMaxScaler()
minmax_scaled = minmax_scaler.fit_transform(df)

print("\nMin-Max Scaled Data:")
print(pd.DataFrame(minmax_scaled, columns=df.columns))

#train-test split

X = df[['study_hours']]
y = df['scores']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain-Test Split:")
print("X_train:\n", X_train)

print("X_test:\n", X_test)

print("y_train:\n", y_train)

print("y_test:\n", y_test)