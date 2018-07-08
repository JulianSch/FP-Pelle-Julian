import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
# import uncertainties.unumpy as unp
import matplotlib.pyplot as plt
f, s1, s2, h1, h2 = np.genfromtxt('messung1.txt', unpack=True)
F, S1, S2, H1, H2 = np.genfromtxt('messung2.txt', unpack=True)
# f = Frequenz in kHz s = Sweepspule, h = Horizontalfeldspule
# 1 Umdrehung entspricht 0.1 V und 0.1 Ampere
v = (24.1 + 8)*0.1*0.1  # Vertikalfeldspule + Umrechnung in Ampere

# Umrechnung in Ampere
h1 = (h1 + 8)*10**(-1)
h2 = (h2 + 8)*10**(-1)
H1 = (H1 + 8)*10**(-1)
H2 = (H2 + 8)*10**(-1)
s1 = (s1 + 8)*10**(-2)
s2 = (s2 + 8)*10**(-2)
S1 = (S1 + 8)*10**(-2)
S2 = (S2 + 8)*10**(-2)
print(h1)
# Vertikalfeldspulenkonstanten
rv = 11.735*10**(-2)  # Radius
lv = 2*np.pi * rv * 20  # Drahtlänge
Nv = 20  # Windungszahl

Bv = const.mu_0*v*Nv/lv
print('')
print('Vertikalfeld =', Bv)
print('')

# Konstanten der Spulen h = Horizontal s = Sweep
rh = 15.79**(-2)  # Radius
lh = 2*np.pi * rh * 154  # Drahtlänge
Nh = 154  # Windungszahl

rs = 16.39*10**(-2)
ls = 2*np.pi * rs * 11
Ns = 11

print('')
print('Erste Messung:')
print('')
Bh1 = const.mu_0*h1*Nh/lh
print('Horizontalfeld Isotop 1 =', Bh1)
Bh2 = const.mu_0*h2*Nh/lh
print('')
print('Horizontalfeld Isotop 2 =', Bh2)

Bs1 = const.mu_0 * s1*Ns/ls
print('Sweepfeld Isotop 2 =', Bs1)
Bs2 = const.mu_0 * s2*Ns/ls
print('')
print('Sweepfeld Isotop 2 =', Bs2)

print('')
print('Zweite Messung:')
print('')
BH1 = const.mu_0*H1*Nh/lh
print('Horizontalfeld Isotop 1 =', BH1)
BH2 = const.mu_0*H2*Nh/lh
print('')
print('Horizontalfeld Isotop 2 =', BH2)
BS1 = const.mu_0 * S1*Ns/ls
print('')
print('Sweepfeld Isotop 1 =', BS1)
BS2 = const.mu_0 * S2*Ns/ls
print('')
print('Sweepfeld Isotop 2 =', BS2)

print('')
print('####################################################################')
print('')


# Berechnung des Erdmagnetfelds

def g(x, a, b):
    return a*x + b

# erstes Isotop
params1, covariance1 = curve_fit(g, f, Bh1)
errors1 = np.sqrt(np.diag(covariance1))
params2, covariance2 = curve_fit(g, f, Bh2)
errors2 = np.sqrt(np.diag(covariance2))
# zweites Isotop
params3, covariance3 = curve_fit(g, f, BH1)
errors3 = np.sqrt(np.diag(covariance3))
params4, covariance4 = curve_fit(g, f, BH2)
errors4 = np.sqrt(np.diag(covariance4))

print('Fitparameter (b entspricht Horizontalfeldkomponente)')
print('')
print('Messung 1:')
print('')
print('ah1 =', params1[0], '+/-', errors1[0])
print('bh1 =', params1[1], '+/-', errors1[1])
print('')
print('ah2 =', params2[0], '+/-', errors2[0])
print('bh2 =', params2[1], '+/-', errors2[1])
print('')
print('Messung 2:')
print('')
print('aH1 =', params3[0], '+/-', errors3[0])
print('bH1 =', params3[1], '+/-', errors3[1])
print('')
print('aH2 =', params4[0], '+/-', errors4[0])
print('bH2 =', params4[1], '+/-', errors4[1])
# Bfeld richtung !? achte auf addition subtraktion
x_plot = np.linspace(0, 1050)
# erstes Isotop
plt.grid()
plt.plot(f, Bh1*1000000, 'rx', label='Isotop 1')
plt.plot(x_plot, g(x_plot, *params1)*1000000, 'b-', label='Fit 1', linewidth=1)
plt.plot(f, Bh2*1000000, 'yo', label='Isotop 2')
plt.plot(x_plot, g(x_plot, *params2)*1000000, 'g-', label='Fit 2', linewidth=1)
plt.legend(loc="best")
plt.savefig("Bfeldfit1.pdf")
plt.clf()
# Zweites Isotop
plt.grid()
plt.plot(f, BH1*1000000, 'rx', label='Isotop 1')
plt.plot(x_plot, g(x_plot, *params3)*1000000, 'b-', label='Fit 1', linewidth=1)
plt.plot(f, BH2*1000000, 'yo', label='Isotop 2')
plt.plot(x_plot, g(x_plot, *params4)*1000000, 'g-', label='Fit 2', linewidth=1)
plt.legend(loc="best")
plt.savefig("Bfeldfit2.pdf")
plt.clf()
# Berechnung der Landé-Faktoren
m1 = ufloat(params1[0], errors1[0])
m2 = ufloat(params2[0], errors2[0])
m3 = ufloat(params3[0], errors3[0])
m4 = ufloat(params4[0], errors4[0])
# print(m1)
# m in Feld pro Frequenz
g1 = const.h/(m1*const.value('Bohr magneton'))
g2 = const.h/(m2*const.value('Bohr magneton'))
g3 = const.h/(m3*const.value('Bohr magneton'))
g4 = const.h/(m4*const.value('Bohr magneton'))

print('')
print('Landé Faktoren:')
print('')
print('erste Messung:')
print('g1 =', g1)
print('g2 =', g2)
print('')
print('zweite Messung:')
print('g3 =', g3)
print('g4 =', g4)
print('')
# Berechnung der Kernspins
print('Kernspins:')
print('')
print('erste Messung:')
I1 = ((2.0023 / g1) - 1)/2
print('I1 =', I1)
I2 = ((2.0023 / g2) - 1)/2
print('I2 =', I2)
print('')
print('zweite Messung:')
I3 = ((2.0023 / g3) - 1)/2
print('I3 =', I3)
I4 = ((2.0023 / g4) - 1)/2
print('I4 =', I4)
