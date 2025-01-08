from sklearn.datasets import load_iris

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Load the dataset
iris = load_iris()

# Create a DataFrame
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
data['species'] = iris.target

# Display the first five rows
print(data.head())



# Pairplot to visualize feature relationships
sns.pairplot(data, hue='species', palette='Set2')
plt.show()
