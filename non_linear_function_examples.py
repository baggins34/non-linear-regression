"""
Non Linear Function Examples
by
?brahim Halil Bayat, PhD
?stanbul Technical University
?stanbul, Turkey

"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab as pl

"""
Linear Regression 

x = np.arange(-5.0, 5.0, 0.1)
y = 2*x + 3
y_noise = np.random.normal(size=x.size)
y_data = y + y_noise
plt.figure(figsize=(8, 6))
plt.plot(x, y_data, 'bo')
plt.plot(x, y, 'r')
plt.ylabel("Dependent Variable")
plt.xlabel("Independent Variable")
plt.show()
"""

x = np.arange(-5.0, 5.0, 0.1)

# As a cubic function
y = (x**3) + (x**2) + x + 3
y_noise = 20*np.random.normal(size=x.size)
y_data = y + y_noise
plt.figure(figsize=(8, 6))
plt.plot(x, y, 'r')
plt.plot(x, y_data, 'bo')
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("As a cubic function")
plt.show()

# Quadratic function
y2 = np.power(x,2)
y_noise2 = 2*np.random.normal(size=x.size)
y_data2 = y2 + y_noise2
plt.figure(figsize=(8, 6))
plt.plot(x, y2, 'r')
plt.plot(x, y_data2, 'bo')
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("As a quadratic function")
plt.show()

# As an exponential function
y3 = np.exp(x)
y_noise3 = 2*np.random.normal(x)
y_data3 = y3 + y_noise3
plt.figure(figsize=(8, 6))
plt.plot(x, y3, 'r')
plt.plot(x, y_data3, 'bo')
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("As exponential")
plt.show()

# As a logarithmic function
y4 = np.log(x)
y_noise4 = np.random.normal(x)
y_data4 = y4 + y_noise4
plt.figure(figsize=(8, 6))
plt.plot(x, y4, 'r')
plt.plot(x, y_data4, 'bo')
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("As Logarithmic")
plt.show()

# As a sigmoid/logistic function
y5 = 1-4/(1+np.power(3, x-2))
y_noise5 = np.random.normal(x)
y_data5 = y5 + y_noise5
plt.figure(figsize=(8, 6))
plt.plot(x, y5, 'r')
plt.plot(x, y_data5, 'bo')
plt.xlabel("Independent Variable")
plt.ylabel("Dependent Variable")
plt.title("As Sigmoid/Logistic")
plt.show()
