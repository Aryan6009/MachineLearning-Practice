import pandas as pd

data = {
    "Name" : ['raman', 'suresh', None, 'anita', 'john'],
    "salary" : [1000, 2000, 1500, None, 3000],
    "age" : [None, 23, None, 29, 31]
}

df = pd.DataFrame(data)

print("Original DataFrame:")
print(df)

#find all the missing values in the dataframe

missing_values = df.isnull().sum()
print("\nMissing Values in DataFrame:\n", missing_values)

#percent of missing values in each column
percent_missing = df.isnull().mean()*100
print("\nPercentage of Missing Values in DataFrame:\n", percent_missing)

#drop rows with missing values
df_drop = df.dropna()
print("\nDataFrame after dropping rows with missing values:")
print(df_drop)


#fill missing values with mean for numerical columns and 'Unknown' for categorical columns
# df['salary'].fillna(df['salary'].mean(), inplace=True)
# df['age'].fillna(df['age'].mean(), inplace=True)

# print(df)