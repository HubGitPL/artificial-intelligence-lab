import numpy as np
import matplotlib.pyplot as plt
import random as rnd

from data import get_data, inspect_data, split_data

data = get_data()
inspect_data(data)

train_data, test_data = split_data(data)


#my fuctions
def mse_calculation(x, y, theta):
    #y= ax+b
    y0=float(theta[0])+float(theta[1])*x
    m = len(x)
    mse = np.sum((y0 - y)**2)/m
    return mse

# Simple Linear Regression
# predict MPG (y, dependent variable) using Weight (x, independent variable) using closed-form solution
# y = theta_0 + theta_1 * x - we want to find theta_0 and theta_1 parameters that minimize the prediction error

# We can calculate the error using MSE metric:
# MSE = SUM (from i=1 to n) (actual_output - predicted_output) ** 2

# get the columns
y_train = train_data['MPG'].to_numpy() #wektor jednowymiarowy
x_train = train_data['Weight'].to_numpy()

y_test = test_data['MPG'].to_numpy()
x_test = test_data['Weight'].to_numpy() # x1 x2

# TODO: calculate closed-form solution
theta_best = [0, 0]
# y=mx+c  // y=ax+b
matrix_X = np.ones_like(x_train)
matrix_X = np.column_stack((matrix_X, x_train))
theta_best = np.linalg.inv(matrix_X.T.dot(matrix_X))
theta_best = theta_best.dot(matrix_X.T).dot(y_train)
print(theta_best)

# TODO: calculate error
mse = mse_calculation(x_train, y_train , theta_best)
print('%.3f' % mse)
mse = mse_calculation(x_test, y_test , theta_best)
print('%.3f' % mse)
# plot the regression line
x = np.linspace(min(x_test), max(x_test), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(x_test, y_test)  #scatter punkty
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
#
# TODO: standardization
# z = (x - mean) / std
std_x = np.std(x_train)
mean_x = np.sum(x_train) / len(x_train)
z_train_x = ((x_train - mean_x) / std_x)
z_test_x = ((x_test - mean_x) / std_x)

std_y = np.std(y_train)
mean_y = np.sum(y_train) / len(y_train)
z_train_y = ((y_train - mean_y) / std_y)
z_test_y = ((y_test - mean_y) / std_y)


# TODO: calculate theta using Batch Gradient Descent
theta_best = [rnd.random(), rnd.random()]
learning_rate = 0.1
iterations_of_learing = 100
matrix_X = np.ones_like(z_train_x)
matrix_X = np.column_stack((matrix_X, z_train_x))
for i in range(iterations_of_learing):
    m = len(z_train_x)
    mse_gradient = (2 / m) * matrix_X.T.dot(matrix_X.dot(theta_best) - z_train_y)
    theta_best = theta_best - mse_gradient * learning_rate
   # print(mse_gradient, theta_best)
#print(theta_best)
# TODO: calculate error
mse_after_gradient = mse_calculation(z_train_x, z_train_y, theta_best)
print('%.3f' % mse_after_gradient)
mse_after_gradient = mse_calculation(z_test_x, z_test_y, theta_best)
print('%.3f' % mse_after_gradient)
#plot the regression line
x = np.linspace(min(z_test_x), max(z_test_y), 100)
y = float(theta_best[0]) + float(theta_best[1]) * x
plt.plot(x, y)
plt.scatter(z_test_x, z_test_y)
plt.xlabel('Weight')
plt.ylabel('MPG')
plt.show()
