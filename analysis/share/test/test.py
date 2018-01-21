#! /usr/bin/env python
import numpy as np
from scipy.optimize import fsolve
from scipy import integrate
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.stattools import adfuller as ADF
a =np.array([2,0,1,5])
print a
print a[:3]
print a.min()
b =np.array([[1,2,3],[4,5,6]])
print b
print b*b

def f(x):
    x1=x[0]
    x2=x[1]
    return [2*x1-x2**2-1,x1**2-x2-2]
result=fsolve(f,[1,1])
print result

def g(x):
    return ((1-x**2)**0.5)
pi_2,err = integrate.quad(g,-1,1)
print pi_2*2

x = np.linspace(0,10,1000)
y = np.sin(x)+1
z = np.cos(x**2)+1

plt.figure(figsize = (8,4))
plt.plot(x, y, label = '$\sin x+1$', color = 'red', linewidth = 2 )
plt.plot(x, z, 'b--', label = '$\cos x^2+1$' )
plt.xlabel('Time(s)')
plt.ylabel('Volt')
plt.title('A Simple Example')
plt.ylim(0,2.2)
plt.legend()
plt.show()

s = pd.Series([1,2,3],index = ['a','b', 'c'])
d = pd.DataFrame([[1,2,3],[4,5,6]],columns = ['a', 'b', 'c'])
d2 = pd.DataFrame(s)
print  d.head()
print  d.describe()
print ADF(np.random.rand(100))
