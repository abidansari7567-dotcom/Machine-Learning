import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

df = pd.read_csv(r"D:\practice\spyder\emp_sal.csv")

x = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg.predict(x), color = 'blue')
plt.title('Linear Regression graph')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

lin_model_pred = lin_reg.predict([[6.5]])
print(lin_model_pred)

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree =5)
x_poly = poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y)

lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly, y)

plt.scatter(x, y, color = 'red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

poly_model_pred = lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
print(poly_model_pred) 

from sklearn.svm import SVR
svr_regressor = SVR(kernel='poly',degree=4, C=5, gamma='auto') 
svr_regressor.fit(x, y)   

svr_pred = svr_regressor.predict([[6.5]])   
print(svr_pred)    


from sklearn.neighbors import KNeighborsRegressor
knn_reg = KNeighborsRegressor(n_neighbors=2, weights='distance', algorithm='brute')
knn_reg.fit(x,y)       

knn_pred = knn_reg.predict([[6.5]])
print(knn_pred)

from sklearn.tree import DecisionTreeRegressor
dt_reg = DecisionTreeRegressor(criterion='poisson', max_depth=3, random_state=0)
dt_reg.fit(x,y)

dt_pred = dt_reg.predict([[6.5]])
print(dt_pred)

from sklearn.ensemble import RandomForestRegressor
rf_reg = RandomForestRegressor(n_estimators=20, random_state=43)
rf_reg.fit(x,y)

rf_pred = rf_reg.predict([[6.5]])
print(rf_pred)





           
                                    
                                    
                                    
                                    
                                    
                                    )