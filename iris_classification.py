from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

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



# Split features and target
X = iris.data  # Features
y = iris.target  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
