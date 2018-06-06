import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import curve_fit
# import uncertainties.unumpy as unp

l, r, c, t = np.genfromtxt('datenko.txt', unpack = True)
x = l-r
y = c/t

plt.grid()
plt.plot(x,y, 'rx', label ='Messwerte')
plt.xlabel(r'Verzögerungszeit in ns')
plt.ylabel(r'Zählrate')
plt.legend(loc='best')
plt.savefig('plot.pdf')
