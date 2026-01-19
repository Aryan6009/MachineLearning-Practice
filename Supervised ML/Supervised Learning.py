# Supervised ML :
# 1. predict numbers : marks----> output continuous

# 2. categories : pass/fail---> fixed category of output

# -------------------------------------------------------------

# Regression ----> linear regression-----> predicting numbers

# 1 - finds patterns in old data
# 2 - straight line
# 3 - predict new values]

# y = mx + b
# -------------------------------------------------------------

# from sklearn.linear_model import LinearRegression

# model = LinearRegression()
# model.fit(X, y)
# model.predict([[value]])

# --------------------------------------------------------------

# example:

from sklearn.linear_model import LinearRegression
model = LinearRegression()
  
hours_studied = [[1], [2], [3], [4], [5]]
marks = [20, 40, 60 ,80 ,95]

model.fit(hours_studied, marks)

hours = float(input("enter the hours you studied for: "))
predicted_marks = model.predict([[hours]])

print(f"based on the number of hours you studied: {hours}, you may score around {predicted_marks}")

# ---------------------------------------------------------------

# Logistic Regression(classification)
# ---->used to predict binary outcomes(yes/no, 0/1)

# ---------------------------------------------------------------

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

X = [[1], [2], [3], [4], [5]] #hours studied input
Y = [0, 0, 1, 1 ,1] #result 0 F, 1 P

model.fit(X,Y)

hours = float(input("enter the hours you studied for: "))
predicted_outcome = model.predict([[hours]])[0]

if predicted_outcome == 1:

    print(f"based on the number of hours you studied: {hours}, you may PASS")

else:
    print(f"based on the number of hours you studied: {hours}, you may FAIL")

# ---------------------------------------------------------------

# K-Nearest Neighbors (KNN) Classification
#---> used for both classification and regression tasks
#---> classifies data points based on the majority class of their nearest neighbors

# ---------------------------------------------------------------

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=3)

X = [[180,2], [200,2.5], [220,3], [250,3.5], [300,4]] #features: weight and engine size                            
Y = ['Sedan', 'Sedan', 'SUV', 'SUV', 'Truck'] #labels: car types
model.fit(X,Y)
weight = float(input("Enter the weight of the car: "))
engine_size = float(input("Enter the engine size of the car: "))
predicted_car_type = model.predict([[weight, engine_size]])[0]
print(f"The predicted car type is: {predicted_car_type}")

# ---------------------------------------------------------------

# Decision Trees
# ---> used for both classification and regression tasks
# ---> splits data into branches to make predictions based on feature values
# eg:- 
#Is Fever > 100.4?
#   Yes ---> Does the patient have a cough?
#       Yes ---> Flu
#       No ---> Common Cold
#   No ---> Healthy

# ---------------------------------------------------------------

from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()

X= [[7,2], [8,3], [9,8], [10,9]] #features: FURIT SIZE AND COLOR SHADE, color shade: 1- RED, 10- ORANGE
Y = [0, 0, 1, 1] #labels: 0- Apple, 1- Orange

model.fit(X,Y)
size = float(input("Enter the size of the fruit: "))
color_shade = float(input("Enter the color shade of the fruit (1-10): "))

predicted_fruit = model.predict([[size, color_shade]])[0]

if predicted_fruit == 0:
    print("The predicted fruit is: Apple")
else:
    print("The predicted fruit is: Orange")