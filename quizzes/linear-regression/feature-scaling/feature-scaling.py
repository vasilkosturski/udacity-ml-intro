import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

# Assign the data to predictor and outcome variables
train_data = pd.read_csv('data.csv', header=None)
X = train_data.iloc[:, :-1]
y = train_data.iloc[:, -1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

lasso_reg = Lasso()
lasso_reg.fit(X_scaled, y)

reg_coef = lasso_reg.coef_

print(reg_coef)
