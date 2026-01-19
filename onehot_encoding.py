import pandas as pd

df = pd.read_csv("language.csv", encoding='latin1')

df_onehot = df.copy()

df_encoded = pd.get_dummies(df_onehot, columns=['Name'])
print("\n one hot encoded dataframe:")
print(df_encoded.head())

#convert encoded values from boolean to binary (0 and 1)

for col in df_encoded.columns:
    if df_encoded[col].dtype == 'bool':
        df_encoded[col] = df_encoded[col].astype(int)
        
print("\n one hot encoded dataframe with binary values:")
print(df_encoded.head())