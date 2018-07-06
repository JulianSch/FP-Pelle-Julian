# import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import curve_fit
# import uncertainties.unumpy as unp

x, y = np.genfromtxt('druckdaten.txt', unpack=True)
x1, y1 = np.genfromtxt('druckdaten2.txt', unpack=True)
x2, y2 = np.genfromtxt('druckdaten3.txt', unpack=True)
x3, y3 = np.genfromtxt('druckdaten4.txt', unpack=True)

L = 100*10**(-3)
lambdavac = 632.99*10**(-9)
data = [x, x1, x2, x3]
for i in data:

    n = 1 + (lambdavac*i)/L
    print(n)
