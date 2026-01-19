import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 40, 45],
    'Spending': [50000, 60000, 70000, 800000, 900000]
}

df = pd.DataFrame(data)

X= df[['Age', 'Spending']]

model = KMeans(n_clusters=2, random_state=42, n_init=10)

df['Group'] = model.fit_predict(X)

plt.figure(figsize=(6,5))
for group in df['Group'].unique():
    group_data = df[df['Group'] == group]
    plt.scatter(group_data['Age'], group_data['Spending'], label=f'Group {group}') #label = Group + str(group)
plt.xlabel('Age')
plt.ylabel('Spending')
plt.title('K-means Clustering of Customers')
plt.legend()
plt.grid(True)
plt.show()
