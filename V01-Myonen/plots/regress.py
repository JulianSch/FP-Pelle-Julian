import matplotlib.pyplot as plt
import numpy as np
# from scipy.optimize import curve_fit
# import uncertainties.unumpy as unp

cha, zeit = np.genfromtxt('datenre.txt', unpack = True)

plt.grid()
plt.plot(cha, zeit, 'rx', label ='Messwerte')
plt.xlabel(r'Verzögerungszeit in ns')
plt.ylabel(r'Zählrate')
plt.legend(loc='best')
plt.savefig('regress.pdf')

def f(cha, zeit, a, b):
