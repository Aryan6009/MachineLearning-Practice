import  pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

#create a sample dataset

data = {
    'Age' : [25, 30, 35, 40, 45, 50],
    'Income' : [30000, 40000, 50000, 60000, 70000, 80000],
    'SpendingScore' : [70,60,50,40,30,20],
    'Savings' : [1000, 5000, 8000, 10000, 15000, 20000]
}

df = pd.DataFrame(data)

# Standardize the data
scalar = StandardScaler()
scaled_data =  scalar.fit_transform(df)

# Apply PCA

pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_data)

pca_df = pd.DataFrame(pca_result, columns =  ['PCA1', 'PCA2'])

explained_variance = pca.explained_variance_ratio_
print("percentage of variance explained by each principal component:")
print(np.round(explained_variance * 100, 2))

# Visualize the PCA result
plt.figure(figsize=(8,6))
plt.scatter(pca_df['PCA1'], pca_df['PCA2'], color='blue', s=80) # s = size of points
plt.title('PCA Result')
plt.xlabel('PCA 1 Main Pattern')
plt.ylabel('PCA 2 Minor Pattern')
plt.grid(True)
plt.show()

print("New data with 2 principal components:")
print(pca_df)