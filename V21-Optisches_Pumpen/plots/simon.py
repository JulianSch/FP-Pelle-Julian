import matplotlib as mpl
mpl.use('pgf')
import numpy as np
import scipy.constants as const
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)
mpl.rcParams.update({
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
    'pgf.texsystem': 'lualatex',
    'pgf.preamble': r'\usepackage{unicode-math}\usepackage{siunitx}'
})

# f, p1, p2, h1, h2 = np.genfromtxt('daten.txt', unpack=True)
# F, P1, P2, H1, H2 = np.genfromtxt('daten2.txt', unpack=True)
Fs, P1s, P2s, H1s, H2s = np.genfromtxt('messung1.txt', unpack=True)
# fs, p1s, p2s, h1s, h2s = np.genfromtxt('daten4.txt', unpack=True)
# iv = 2.24*0.1
# i1 = p1*0.1
# i2 = p2*0.1
# i12 = h1*0.3
# i22 = h2*0.3
# I1 = P1*0.1
# I2 = P2*0.1
# I12 = H1*0.3
# I22 = H2*0.3
iv = (24.1 + 8)*0.1*0.1*0.1  # Vertikalfeldspule + Umrechnung in Ampere

# h1 = (h1 + 8)*10**(-3)
# h2 = (h2 + 8)*10**(-3)

# s1 = (s1 + 8)*10**(-2)
# s2 = (s2 + 8)*10**(-2)

rv = 11.735*10**(-2)  # Radius
lv = 2*np.pi * rv * 20  # Drahtlänge
Nv = 20  # Windungszahl

I1s = P1s*0.1
# I2s = P2s*0.1
I12s = H1s*0.3
# I22s = H2s*0.3
# i1s = p1s*0.1
# i2s = p2s*0.1
# i12s = h1s*0.3
# i22s = h2s*0.3
mu0 = const.mu_0
Bv = mu0*(8/np.sqrt(125))*(20/0.11735)*iv
# B1 = mu0*(8/np.sqrt(125))*(11/0.1639)*i1 + mu0*(8/np.sqrt(125))*(154/0.1579)*i12
# B2 = mu0*(8/np.sqrt(125))*(11/0.1639)*i2 + mu0*(8/np.sqrt(125))*(154/0.1579)*i22
# B12 = mu0*(8/np.sqrt(125))*(11/0.1639)*I1 + mu0*(8/np.sqrt(125))*(154/0.1579) * I12
# B22 = mu0*(8/np.sqrt(125))*(11/0.1639)*I2 + mu0*(8/np.sqrt(125))*(154/0.1579)*I22
B12s = mu0*(8/np.sqrt(125))*(11/0.1639)*I1s + mu0*(8/np.sqrt(125))*(154/0.1579)*I12s
# B22s = mu0*(8/np.sqrt(125))*(11/0.1639)*I2s + mu0*(8/np.sqrt(125))*(154/0.1579)*I22s
# B1s = mu0*(8/np.sqrt(125))*(11/0.1639)*i1s + mu0*(8/np.sqrt(125))*(154/0.1579)*i12s
# B2s = mu0*(8/np.sqrt(125))*(11/0.1639)*i2s + mu0*(8/np.sqrt(125))*(154/0.1579)*i22s

Bdiff = mu0*(8/np.sqrt(125))*(154/0.1579)*0.0255
print('Bdiff =', Bdiff)
# print('B1 =', B1)
# print('B2 =', B2)
# print('B12 =', B12)
# print('B22 =', B22)
print('B12s =', B12s)
# print('B22s =', B22s)
# print('B1s =', B1s)
# print('B2s =', B2s)

print('Vertikalfeld =', Bv)


def l(x, a, b):
    return a*x+b

# params, covariance = curve_fit(l, Fs, B1s)
# errors = np.sqrt(np.diag(covariance))
# params2, covariance2 = curve_fit(l, Fs, B2s)
# errors2 = np.sqrt(np.diag(covariance2))
params3, covariance3 = curve_fit(l, Fs, B12s)
errors3 = np.sqrt(np.diag(covariance3))
# params4, covariance4 = curve_fit(l, Fs, B22s)
# errors4 = np.sqrt(np.diag(covariance4))

# print('a =', params[0], '±', errors[0])
# print('b =', params[1], '±', errors[1])
# print('a2 =', params2[0], '±', errors2[0])
# print('b2 =', params2[1], '±', errors2[1])
print('A =', params3[0], '±', errors3[0])
print('B =', params3[1], '±', errors3[1])
# print('A2 =', params4[0], '±', errors4[0])
# print('B2 =', params4[1], '±', errors4[1])

x_plot = np.linspace(-10, 1050)

# plt.plot(F, B1*1000000, 'rx', label='Messwerte')
# plt.plot(F, B2*1000000, 'bx', label='Messwerte 2')
plt.plot(Fs, B12s*100000, 'ro', label=r'Werte korrigiert um $\SI{25.5}{\milli\ampere}$')
# plt.plot(Fs, B22*1000000, 'bo', label=r'Werte 2 korrigiert um $\SI{25.5}{\milli\ampere}$')
# plt.plot(x_plot, l(x_plot, *params)*1000000, 'g--', label='Fit1', linewidth=1)
# plt.plot(x_plot, l(x_plot, *params2)*1000000, 'y--', label='Fit2', linewidth=1)
plt.plot(x_plot, l(x_plot, *params3)*100000, 'g-', label='Fit3', linewidth=1)
# plt.plot(x_plot, l(x_plot, *params4)*1000000, 'y-', label='Fit4', linewidth=1)
plt.legend(loc="best")
plt.xlabel(r'Frequenz in $\si{\kilo\hertz}$')
plt.ylabel(r'$B_m + B_{Erde}$ in $\si{\micro\tesla}$')
plt.tight_layout
plt.savefig("Plot2.pdf")
