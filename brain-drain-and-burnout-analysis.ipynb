import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data from a CSV file
data = pd.read_csv('employee_data.csv')

# Explore the data
print(data.head())
print(data.describe())

# Correlation matrix
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True)
plt.show()

# Regression analysis (example using Linear Regression)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = data[['workload', 'job_satisfaction']]
y = data['burnout_score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
