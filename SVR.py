import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv(r"D:\practice\spyder\emp_sal.csv")
x = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.svm import SVR
regressor = SVR(kernel= 'poly', degree=5)
regressor.fit(x, y)

y_pred = regressor.predict([[6.5]])

plt.scatter(x, y, color = 'red')
plt.plot(x, regressor.predict(x), color = 'blue')
plt.title('Truth of Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()