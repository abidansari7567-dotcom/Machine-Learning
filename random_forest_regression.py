import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv(r"D:\practice\spyder\emp_sal.csv")
x = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=15)
regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])

x_grid = np.arange(min(x), max(x), 0.01)
X_grid = x_grid.reshape((len(x_grid),1))
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color = 'blue')
plt.title('Truth or Bluff (random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()