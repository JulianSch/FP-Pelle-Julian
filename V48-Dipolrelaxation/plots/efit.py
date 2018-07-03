import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.stats as stats
from scipy.optimize import curve_fit
# from uncertainties import correlated_values, correlation_matrix
Te, Ie = np.genfromtxt('daten1.txt', unpack=True)

Te = (Te+273.15)  # *const.value('Boltzmann constant')
Ie = -Ie
diff = np.array([])
for i in range(len(Te)-1):  # geht Länge des Arrays durch
    d = Te[i]-Te[i+1]
    diff = np.append(diff, d)

print(diff)

print(np.mean(diff))
print(np.std(diff, ddof=1)/len(diff)**(1/2))
print(stats.sem(diff))


def f(x, a, m):
    return a + m*x
# a entspricht nem Haufen Konstanten und W der Aktivierungsarbeit
params, cov = curve_fit(f, 1/Te, np.log(Ie), p0=(10, 10))
print(cov)
print(params[1]*const.k)
# params = correlated_values(params, cov)
for p in params:
    print(p)
Te_plot = np.linspace(235, 265, 1000)

plt.plot(Te, Ie, 'rx', label='Messwerte')
plt.plot(Te_plot, np.exp(f(1/Te_plot, *params)), 'b-', label='Fit')
plt.xlabel('T in K')
plt.ylabel('I in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('efit.pdf')
