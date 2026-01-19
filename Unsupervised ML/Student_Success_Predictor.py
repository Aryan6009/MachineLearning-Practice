import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load the dataset
df = pd.read_csv("Unsupervised ML\student_success_dataset.csv")

# Display the first few rows of the dataset

print("First few rows of the dataset:")
print(df.head())

print("Dataset info:")
print(df.info())

print("Dataset Shape:")
print(f'Rows: {df.shape[0]}, Columns: {df.shape[1]}') #0 and 1 used to access rows and columns respectively because shape returns a tuple

print("summary statistics of the dataset:")
print(df.describe(include='all'))

print("Checking for missing values:")
print(df.isnull().sum())

# Data Preprocessing

le = LabelEncoder()
df['Internet'] =le.fit_transform(df['Internet'])
df['Passed'] = le.fit_transform(df['Passed']) # yes and no to 1 and 0

print("after encoding:")
print(df.head())

print('dtypes after encoding:')
print(df.dtypes)

# Feature Scaling

features = ['StudyHours', 'Attendance', 'PastScore', 'SleepHours']
df_scaled = df.copy()

scaler = StandardScaler()
df_scaled[features] = scaler.fit_transform(df[features])

# train-test split

X = df_scaled[features]
y = df_scaled['Passed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#test9 size of 0.2 means 20% of data will be used for testing and 80% for training

# Training the model

model = LogisticRegression()
model.fit(X_train, y_train)

# Making predictions

y_pred = model.predict(X_test)

# Evaluating the model

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=['Failed' ,'Passed'], yticklabels=['Failed','Passed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

print("------Predict your result------")
try:
    study_hours = float(input("Enter average study hours per week: "))
    attendance = float(input("Enter attendance percentage: "))
    past_score = float(input("Enter past exam score: "))
    sleep_hours = float(input("Enter average sleep hours per night: "))

    user_data = pd.DataFrame([{
        'StudyHours': study_hours,
        'Attendance': attendance,
        'PastScore': past_score,
        'SleepHours': sleep_hours
    }])
    # converted to dataframe to match the model input format
    user_input_scaled = scaler.transform(user_data)
    
    prediction = model.predict(user_input_scaled)[0]
    
    result = 'Passed' if prediction == 1 else 'Failed'
    
    print(f"Based on the input data, the predicted result is: {result}")
    
except Exception as e:
    print("Error in input. Please enter valid numerical values.")
    