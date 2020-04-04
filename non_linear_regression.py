"""
Non Linear Regression
by
Ibrahim Halil Bayat, PhD
Istanbul Technical University
Istanbul, Turkey

"""
import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt

df = pd.read_csv("china_gdp.csv")
plt.figure(figsize=(8, 6))
x_data, y_data = (df['Year'].values, df['Value'].values)
plt.plot(x_data, y_data, 'ro')
plt.xlabel("Year")
plt.ylabel("The Growth")
plt.title("Year vs Growth")
plt.show()

def sigmoid(x, beta1, beta2):
    y = 1 / (1 + np.exp(-beta1*(x-beta2)))
    return y

beta1 = 0.10
beta2 = 1990.0
y_pred = sigmoid(x_data, beta1,beta2)*15000000000000
plt.figure(figsize=(8, 6))
plt.plot(x_data, y_pred, 'g')
plt.plot(x_data, y_data, 'ro')
plt.xlabel("Year")
plt.ylabel("The Growth")
plt.title("Year vs Growth with Sigmoid")
plt.show()

# Time to normalize our data
xdata = x_data/(max(x_data))
ydata = y_data/(max(y_data))
plt.figure(figsize=(8, 6))
plt.plot(x_data, y_pred, 'g')
plt.plot(x_data, y_data, 'ro')
plt.xlabel("Year")
plt.ylabel("The Growth")
plt.title("Year vs Growth Normalized")
plt.show()

from scipy.optimize import curve_fit as cf
popt, pcov = cf(sigmoid, xdata, ydata)
print(" beta1 = %.4f, beta2 = %.4f" % (popt[0], popt[1]))
print(popt)
print(pcov)


plt.figure(figsize=(8,5))
y = sigmoid(xdata, *popt)
plt.plot(xdata, ydata, 'ro', label='data')
plt.plot(xdata, y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

msk = np.random.rand(len(df)) < 0.8
xtrain = xdata[msk]
ytrain = ydata[msk]
xtest = xdata[~msk]
ytest = ydata[~msk]

popt, pcov = cf(sigmoid, xtrain, ytrain)

y_predicted = sigmoid(xtest, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_predicted - ytest)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_predicted - ytest) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(y_predicted, ytest))