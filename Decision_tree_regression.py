import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"D:\practice\spyder\emp_sal.csv")
x = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion='friedman_mse', splitter='random')
regressor.fit(x,y)

from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators=300, random_state=0)
reg.fit(x,y)

y_pred = reg.predict([[6.5]])

plt.scatter(x,y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth or bluff (Decission tree Regressiom)')
plt.xlabel('Position Level')
plt.ylabel('salary')
plt.show()


x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, regressor.predict(x_grid), color ='blue')
plt.title('Truth or Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()