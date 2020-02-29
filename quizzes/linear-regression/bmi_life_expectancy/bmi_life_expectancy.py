import pandas as pd
from sklearn.linear_model import LinearRegression

bmi_life_data = pd.read_csv("data.csv")

# Make and fit the linear regression model
bmi_life_model = LinearRegression()
bmi = bmi_life_data[['BMI']]
life_expectancy = bmi_life_data[['Life expectancy']]
bmi_life_model.fit(bmi, life_expectancy)

# Make a prediction using the model
laos_life_exp = bmi_life_model.predict([[21.07931]])