from sklearn.preprocessing import LabelEncoder
import pandas as pd

df = pd.read_csv("Datasets\language.csv",encoding='latin1')
df_label = df.copy()

le = LabelEncoder()

df_label['Name_encoded'] = le.fit_transform(df_label['Name'])
df_label['Code_encoded'] = le.fit_transform(df_label['Code'])

print("\n label encoded dataframe:")
print(df_label[['Name', 'Name_encoded', 'Code', 'Code_encoded']].head())