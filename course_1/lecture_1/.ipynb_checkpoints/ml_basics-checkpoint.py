from sklearn.datasets import fetch_california_housing
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

# Load the housing prices data
housing = fetch_california_housing()

X = housing.data
y = housing.target

print('\n\t Shape of the features table:', X.shape)
print('\n\t Length of the targets column:', y.shape)

# Build DataFrame (table) with feature and target
df = pd.DataFrame(X, columns = housing.feature_names)
df['Target'] = y
print('\n', df.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.50)

m1 = LinearRegression()
m2 = DecisionTreeRegressor()

m1.fit(X_train, y_train)
m2.fit(X_train, y_train)

p1_test = m1.predict(X_test)
p2_test = m2.predict(X_test)

df_test = pd.DataFrame(X_test, columns = housing.feature_names)
df_test['Target'] = y_test
df_test['Preds_1'] = p1_test
df_test['Preds_2'] = p2_test
print('\n', df_test.head())
