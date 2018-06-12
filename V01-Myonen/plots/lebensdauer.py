import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat

counts = np.genfromtxt('lebensdauer.txt')

def e(t, a, b, c):
    return a*np.exp(-t/b)+c

t = np.linspace(0,399,400)
print(t)

params , cov = curve_fit(e , t , counts)

plt.plot(counts, 'y.', label = 'Messwerte')
plt.plot(e(t, *params),t, 'b-', label = 'fit')

params = correlated_values(params, cov)
for p in params:
    print(p)

plt.grid()
plt.xlabel('Kanal')
plt.ylabel('Counts')


plt.legend(loc='best')
plt.savefig('lebensdauer.pdf')
plt.clf()
#logcounts = np.log(counts)
