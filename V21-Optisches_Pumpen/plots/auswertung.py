import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import ufloat
# import uncertainties.unumpy as unp
import matplotlib.pyplot as plt

# Einlesen der Messwerte
f, s1, s2, h1, h2 = np.genfromtxt('messung1.txt', unpack=True)
F, S1, S2, H1, H2 = np.genfromtxt('messung2.txt', unpack=True)
# f = Frequenz in kHz s = Sweepspule, h = Horizontalfeldspule

# Umrechnung in Ampere

v = (24.1 + 8)*0.1*0.1  # Vertikalfeldspule

h1 = (h1)*10**(-2)*0.3  # zwei Größenodrunungn aus den Umdrehungen (Laborbuch)
h2 = (h2)*10**(-2)*0.3  # 0.3 aus der Anleitung
H1 = (H1)*10**(-2)*0.3
H2 = (H2)*10**(-2)*0.3
s1 = (s1)*10**(-2)  # eine Größenordnung aus den Umdrehungen (Laborbuch)
s2 = (s2)*10**(-2)  # zweite aus der Anleitungen
S1 = (S1)*10**(-2)
S2 = (S2)*10**(-2)

print('Stromstärken:')
print()

print('Horizontalfeld1 Messung 1: ', h1)
print('Sweepfeld1 Messung 1: ', s1)
print('Horizontalfeld2 Messung 1: ', h2)
print('Sweepfeld2 Messung 1: ', s2)
print('Horizontalfeld1 Messung 2: ', H1)
print('Sweepfeld1 Messung 2: ', S1)
print('Horizontalfeld2 Messung 2: ', H2)
print('Sweepfeld2 Messung 2: ', S2)

# Vertikalfeldspulenkonstanten
rv = 11.735*10**(-2)  # Radius
lv = 2*np.pi * rv * 20  # Drahtlänge
Nv = 20  # Windungszahl

Bv = const.mu_0*(8*v*Nv)/((np.sqrt(125))*rv)
print('')
print('Vertikalfeld =', Bv*10**3)
print('')

# Konstanten der Spulen h = Horizontal s = Sweep
rh = 15.79*10**(-2)  # Radius
lh = 2*np.pi * rh * 154  # Drahtlänge
Nh = 154  # Windungszahl

rs = 16.39*10**(-2)
ls = 2*np.pi * rs * 11
Ns = 11

print('')
print('Erste Messung:')
print('')

# Magnetfelder erste Messung
Bh1 = const.mu_0*(8*h1*Nh)/((np.sqrt(125))*rh)
Bh2 = const.mu_0*(8*h2*Nh)/((np.sqrt(125))*rh)
Bs1 = const.mu_0*(8*s1*Ns)/((np.sqrt(125))*rs)
Bs2 = const.mu_0*(8*s2*Ns)/((np.sqrt(125))*rs)

print('Summe der Horizontalfelder Messung 1:')
sum1 = Bh1+Bs1
sum2 = Bh2+Bs2
print(sum1)
print(sum2)

print('')
print('Zweite Messung:')
print('')

# Magnetfelder zweite Messung
BH1 = const.mu_0*(8*H1*Nh)/((np.sqrt(125))*rh)
BH2 = const.mu_0*(8*H2*Nh)/((np.sqrt(125))*rh)
BS1 = const.mu_0*(8*S1*Ns)/((np.sqrt(125))*rs)
BS2 = const.mu_0*(8*S2*Ns)/((np.sqrt(125))*rs)

print('Summen der Horizontalfelder Messung2:')
Sum1 = BH1+BS1
Sum2 = BH2+BS2
print(Sum1)
print(Sum2)

print('')
print('####################################################################')
print('')


# Berechnung des Erdmagnetfelds

def g(x, m, b):
    return m*x + b

# erstes Isotop
params1, covariance1 = curve_fit(g, f, sum1)
errors1 = np.sqrt(np.diag(covariance1))
params2, covariance2 = curve_fit(g, f, sum2)
errors2 = np.sqrt(np.diag(covariance2))
# zweites Isotop
params3, covariance3 = curve_fit(g, F, Sum1)
errors3 = np.sqrt(np.diag(covariance3))
params4, covariance4 = curve_fit(g, F, Sum2)
errors4 = np.sqrt(np.diag(covariance4))

print('Fitparameter (b entspricht Horizontalfeldkomponente)')
print('')
print('Messung 1:')
print('')
print('mh1 =', params1[0], '+/-', errors1[0])
print('bh1 =', params1[1], '+/-', errors1[1])
print('')
print('mh2 =', params2[0], '+/-', errors2[0])
print('bh2 =', params2[1], '+/-', errors2[1])
print('')
print('Messung 2:')
print('')
print('mH1 =', params3[0], '+/-', errors3[0])
print('bH1 =', params3[1], '+/-', errors3[1])
print('')
print('mH2 =', params4[0], '+/-', errors4[0])
print('bH2 =', params4[1], '+/-', errors4[1])
# Bfeld richtung !? achte auf addition subtraktion
x_plot = np.linspace(0, 1050)
# erstes Isotop
plt.grid()
plt.plot(f, (sum1)*1000000, 'rx', label='Isotop 1')
plt.plot(x_plot, g(x_plot, *params1)*1000000, 'b-', label='Fit 1', linewidth=1)
plt.plot(f, (sum2)*1000000, 'yo', label='Isotop 2')
plt.plot(x_plot, g(x_plot, *params2)*1000000, 'g-', label='Fit 2', linewidth=1)
plt.legend(loc="best")
plt.xlabel(r'$\nu$ in Hz')
plt.ylabel(r'B in $\mu$T')
plt.savefig("Bfeldfit1.pdf")
plt.clf()
# Zweites Isotop
plt.grid()
plt.plot(F, Sum1*1000000, 'rx', label='Isotop 1')
plt.plot(x_plot, g(x_plot, *params3)*1000000, 'b-', label='Fit 1', linewidth=1)
plt.plot(F, Sum2*1000000, 'yo', label='Isotop 2')
plt.plot(x_plot, g(x_plot, *params4)*1000000, 'g-', label='Fit 2', linewidth=1)
plt.legend(loc="best")
plt.xlabel(r'$\nu$ in Hz')
plt.ylabel(r'B in $\mu$T')
plt.savefig("Bfeldfit2.pdf")
plt.clf()

# Berechnung der Landé-Faktoren
m1 = ufloat(params1[0], errors1[0])
m2 = ufloat(params2[0], errors2[0])
m3 = ufloat(params3[0], errors3[0])
m4 = ufloat(params4[0], errors4[0])

# m in Feld pro Frequenz
g1 = const.h/(m1*const.value('Bohr magneton'))*1000
g2 = const.h/(m2*const.value('Bohr magneton'))*1000
g3 = const.h/(m3*const.value('Bohr magneton'))*1000
g4 = const.h/(m4*const.value('Bohr magneton'))*1000

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
I1 = ((2.0023 / g1) - 1)/2
I2 = ((2.0023 / g2) - 1)/2

print('Kernspins:')
print('')
print('erste Messung:')
print()
print('I1 =', I1)
print('I2 =', I2)
print('')

I3 = ((2.0023 / g3) - 1)/2
I4 = ((2.0023 / g4) - 1)/2

print('zweite Messung:')
print()
print('I3 =', I3)
print('I4 =', I4)
print()

# quadratischer Zeeman-Effekt

mub = 5.7883818012*10**(-24)


def Uq(gF, B, MF, Ehy):
    return ((gF**2 * B**2 * mub**2) * (1-2*MF)/(Ehy))

# Angaben in Joule
EhyA = 4.53*10**(-24)  # * 6*10**(18)
EhyB = 2.01*10**(-24)  # * 6*10**(18)

# obere Grenze
Uq1A = Uq(g1, sum1[9], 2, EhyA)
Uq1B = Uq(g2, sum2[9], 3, EhyB)
Uq2A = Uq(g3, Sum1[9], 2, EhyA)
Uq2B = Uq(g4, Sum2[9], 3, EhyB)

print()
print('quadratisch oben:')
print()
print(Uq1A)
print(Uq1B)
print(Uq2A)
print(Uq2B)
print()


def Ul(gF, B):
    return gF*mub*B

# obere Grenze
Ul1A = Ul(g1, sum1[9])
Ul1B = Ul(g2, sum2[9])
Ul2A = Ul(g3, Sum1[9])
Ul2B = Ul(g4, Sum2[9])

# Essensausgabe
# hoffentlich in Joule

print()
print('linear oben:')
print()
print(Ul1A)
print(Ul1B)
print(Ul2A)
print(Ul2B)
print()
