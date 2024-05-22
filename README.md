import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(0)

# Generate dummy data
data = {
    'Sales': np.random.randint(100, 200, size=100),
    'Competition_Price': np.random.uniform(10, 20, size=100),
    'Distribution': np.random.randint(50, 100, size=100),
    'Advertisement_Spend': np.random.uniform(5000, 20000, size=100)
}

df = pd.DataFrame(data)

# Display the first few rows of the dataframe
print(df.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Split the data into features and target variable
X = df[['Competition_Price', 'Distribution', 'Advertisement_Spend']]
y = df['Sales']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create a linear regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Display the coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)
