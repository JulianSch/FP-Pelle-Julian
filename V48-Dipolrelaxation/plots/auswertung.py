import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
# import scipy.stats as stats
from scipy.optimize import curve_fit
from uncertainties import correlated_values  # , correlation_matrix
from uncertainties.unumpy import (nominal_values as noms)  # ,std_devs as stds)
# from scipy.integrate import quad

T1, I1 = np.genfromtxt('daten2.txt', unpack=True)
T2, I2 = np.genfromtxt('daten3.txt', unpack=True)
t1, i1 = np.genfromtxt('ignoredaten2.txt', unpack=True)
t2, i2 = np.genfromtxt('ignoredaten3.txt', unpack=True)

# systematischer Untergrund Abzug
I1 -= -0.12
I2 -= -0.15
i1 -= -0.12
i2 -= -0.15

T1 = T1+273.15
I1 = -I1
t1 = t1+273.15
i1 = -i1

diff = np.array([])
for i in range(len(T1)-1):  # geht länge des arrays durch
    d = T1[i]-T1[i+1]
    diff = np.append(diff, d)

T2 = T2+273.15
I2 = -I2
t2 = t2+273.15
i2 = -i2
diff = np.array([])

for i in range(len(T2)-1):  # geht länge des arrays durch
    d = T2[i]-T2[i+1]
    diff = np.append(diff, d)


def depol(temp, W, A, a):
    return a+A*np.exp(-W/(temp))  # Funktion für Untergrundfit

params1, cov1 = curve_fit(depol, t1, i1)
Params1 = correlated_values(params1, cov1)

y1 = depol(T1, *noms(Params1))
f1 = I1-y1

plt.plot(T1, f1, 'b.', label='Erste Messreihe')
# plt.plot(t1, depol(t1, *params1), 'r-', label='Untergrund')
plt.plot()
plt.xlabel('T in K')
plt.ylabel('I in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('ohneuntergrund.pdf')
plt.clf()

t1 = np.linspace(220, 340)
plt.plot(T1, I1, 'b.', label='Erste Messreihe')
plt.plot(t1, depol(t1, *params1), 'r-', label='Untergrund')
plt.plot()
plt.xlabel('T in K')
plt.ylabel('I in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('mituntergrund.pdf')
plt.clf()

params2, cov2 = curve_fit(depol, t2, i2)
# print(params2)
Params2 = correlated_values(params2, cov2)

y2 = depol(T2, *noms(Params2))
f2 = I2-y2

plt.plot(T2, f2, 'b.', label='Erste Messreihe')
# plt.plot(t1, depol(t1, *params1), 'r-', label='Untergrund')
plt.plot()
plt.xlabel('T in K')
plt.ylabel('I in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('ohneuntergrund2.pdf')
plt.clf()

t2 = np.linspace(180, 340)
plt.plot(T2, I2, 'b.', label='Zweite Messreihe')
plt.plot(t2, depol(t2, *params2), 'r-', label='Untergrund')
plt.xlabel('T in K')
plt.ylabel('I in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('mituntergrund2.pdf')
plt.clf()

# Anlaufkurvenspaß

print()
print('Strom1= ', f1)
print()


def f(x, a, b):
    return a*x + b

Ta1 = np.array([])
Ia1 = np.array([])
for i in range(3, 8):
    Ta1 = np.append(Ta1, T1[i])
    Ia1 = np.append(Ia1, f1[i])
Ia1 = np.log(Ia1)
Ta1 = 1/Ta1
# Ta1 = np.log(Ta1)
# Ia1[0] = 0.001
paramsa1, cova1 = curve_fit(f, Ta1, Ia1)
print(Ta1, Ia1)
print(paramsa1)

Ta2 = np.array([])
Ia2 = np.array([])
for i in range(16, 31):
    Ta2 = np.append(Ta2, T2[i])
    Ia2 = np.append(Ia2, f2[i])
Ta2 = 1/Ta2
Ia2 = np.log(Ia2)
# Ta2 = np.log(Ta2)
paramsa2, cova2 = curve_fit(f, Ta2, Ia2)
# Paramsa2 = correlated_values(paramsa2, cova2)
print('############', Ta2)
print(Ia2)
print(len(Ta1))
print(len(Ia1))
# Arbeiten
W3 = -paramsa1[0] * const.k
W4 = -paramsa2[0] * const.k
print('Arbeiten')
print(W3)

Ta1_plot = np.linspace(1/245, 1/265, 1000)
plt.plot(Ta1, Ia1, 'rx', label='Messwerte')
plt.plot(Ta1_plot, f(Ta1_plot, *paramsa1), 'b-', label='Ausgleich')
plt.xlabel('1/T in 1/K')
plt.ylabel('ln(I) in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('anlauf1.pdf')
plt.clf()

Ta2_plot = np.linspace(1/235, 1/255, 100)

plt.plot(Ta2, Ia2, 'rx', label='Messwerte')
plt.plot(Ta2_plot, f(Ta2_plot, *paramsa2), 'b-', label='Ausgleich')
plt.xlabel('1/T in 1/K')
plt.ylabel('ln(I) in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('anlauf2.pdf')
plt.clf()

##################################################################
# Integralpart kommt ab hier
##################################################################


def riemann(temp, strom, b, e):
    a = 0
    a1 = 0
    for i in range(b, e, 1):
        t = (temp[i] - temp[i+1])
        s = (strom[i] + strom[i+1])/2  # Riemann Integration
        a += np.sqrt((s*t)**2)
        a1 += s*t
    return a


def lnint(T, yt, H):
    array = np.array([])
    for t in T:
        array = np.append(array, np.trapz(yt[T >= t], T[T >= t]) / (yt[T == t] * H))
    return array

# def lnint(T, yt):
#     array = np.array([])
#     for t in T:
#         array = np.append(array, np.trapz(yt[T >= t], T[T >= t]))
#     return array

# chop of negatives and integrate:
fc1 = np.array([])
Tc1 = T1[(T1 > 240) & (T1 < 275)]
for i in range(1, 11):
    fc1 = np.append(fc1, f1[i])

int1 = lnint(Tc1, fc1, 2)
int1 = int1[int1 > 0]
int1 = np.log(int1)  # logarithmieren
inv1 = 1/Tc1
# print(len(inv1))
inv1c = np.array([])
for i in range(1, 9):
    inv1c = np.append(inv1c, inv1[i])

# print('#############')
# print(len(inv1c))
# print(inv1c)
# print('#############')

fc2 = np.array([])
Tc2 = T2[(T2 > 239) & (T2 < 263)]
for i in range(20, 42):
    fc2 = np.append(fc2, f2[i])

int2 = lnint(Tc2, fc2, 1.26)
int2 = int2[int2 > 0]
inv2 = 1/Tc2
int2 = np.log(int2)

inv2c = np.array([])
for i in range(0, 20):
    inv2c = np.append(inv2c, inv2[i])
print(len(inv2c))

# Ausgabeblock
# print()
# print('Arraylängen Tc, fc, inv, int, je 1 und 2')
# print()
# print('Tc1, fc1')
# print(len(Tc1))
# print(len(fc1))
# print()
# print('Tc2, fc2')
# print(len(Tc2))
# print(len(fc2))
# print()
# print('inv1, int1')
# print(len(inv1))
# print(len(int1))
# print()
# print('inv2, int2')
# print(len(inv2))
# print(len(int2))
# print()
#
# print('Arrays erste Messung: ')
# print()
# print(inv1c)
# print()
#
# print(fc1)
# print()
#
# print(noms(int1))
# print()
#
# print('Arrays zweite Messung: ')
# print()
# print(Tc2)
# print()
#
# print(fc2)
# print()
#
# print(noms(int2))
# print()

# Fits, Plots und RocknRoll


def lin(x, m, b):
    return m * x + b

params1, covariance1 = curve_fit(lin, inv1c, int1)
errors1 = np.sqrt(np.diag(covariance1))

params2, covariance2 = curve_fit(lin, inv2c, int2)
errors2 = np.sqrt(np.diag(covariance2))
# print()
# print('Fitparameter')
# print()
# print(params1)
# print(errors1)
# print()
# print(params2)
# print(errors2)
# params1, cov1 = np.polyfit(inv1, int1, deg=1, cov=True)
# errors = np.sqrt(np.diag(cov1))
#
# print('a = {:.3f} ± {:.3f}'.format(params1[0], errors[0]))
# print('b = {:.3f} ± {:.3f}'.format(params1[1], errors[1]))
#
# params2, cov2 = np.polyfit(inv2, int2, deg=1, cov=True)
# errors2 = np.sqrt(np.diag(cov2))
#
# print('a = {:.3f} ± {:.3f}'.format(params2[0], errors2[0]))
# print('b = {:.3f} ± {:.3f}'.format(params2[1], errors2[1]))

# x_plot1 = np.linspace(0.003, 0.005)
#
# plt.plot(inv1c, int1, 'b.', label='erste Messreihe')
# plt.plot(inv1c, lin(inv1c, *params1), 'r-', label='Lineare Regression')
# plt.xlabel(r'$\frac{1}{T}$ in 1/K')
# plt.ylabel('f(T)')
# plt.grid()
# plt.legend(loc='best')
# plt.savefig('integralplot1.pdf')
# plt.clf()
# # intval_1 = riemann(T1, f1, 1, 40)
# # print('Integralwert = ', intval_1)
#
# plt.plot(inv2c, int2, 'b.', label='zweite Messreihe')
# plt.plot(inv2c, lin(inv2c, *params2), 'r-', label='Lineare Regression')
# plt.xlabel(r'$\frac{1}{T}$ in 1/K')
# plt.ylabel('f(T)')
# plt.grid()
# plt.legend(loc='best')
# plt.savefig('integralplot2.pdf')
# plt.clf()

# Arbeiten

W1 = const.Boltzmann * params1[0]  # * 6.242*10**(18)
W2 = const.Boltzmann * params2[0]  # * 6.242*10**(18)
print()
print('Aktivierungsarbeiten:')
print('W1 = ', W1 * 6.242*10**(18))
print('W2 = ', W2 * 6.242*10**(18))
print()
print('Arbeiten aus der Anlaufkurve:')
W3 = W3   # * 6.242*10**(18)
W4 = W4   # * 6.242*10**(18)
print(W3 * 6.242*10**(18))
print(W4 * 6.242*10**(18))
print()

# Relaxationszeiten
H1 = 3.2  # Heizraten
H2 = 1.26

print(np.max(f1))
Tmax1 = T1[7]  # Maxima
Tmax2 = T2[31]

print('Maxima der Kurven')
print()
print('Tmax1 = ', Tmax1)
print('Tmax2 = ', Tmax2)
print()
taumax1 = (Tmax1*Tmax1)*const.Boltzmann/(W1*H1)
taumax2 = (Tmax2*Tmax2)*const.Boltzmann/(W2*H2)
taumax3 = (Tmax1*Tmax1)*const.Boltzmann/(W3*H1)
taumax4 = (Tmax2*Tmax2)*const.Boltzmann/(W4*H2)

print('Berechnete Werte nach Formel 10:')
print('taumax1 = ', taumax1)
print('taumax2 = ', taumax2)
print('taumax3 = ', taumax3)
print('taumax4 = ', taumax4)
print()
print('Arbeiten')
print(W1, W2, W3, W4)
tau01 = taumax1*np.exp(-W1/(const.Boltzmann*Tmax1))
tau02 = taumax2*np.exp(-W2/(const.Boltzmann*Tmax2))
tau03 = taumax3*np.exp(-W3/(const.Boltzmann*Tmax1))
tau04 = taumax4*np.exp(-W4/(const.Boltzmann*Tmax2))

print('Relaxationszeiten')
print('tau01 = ', tau01)
print('tau02 = ', tau02)
print('tau03 = ', tau03)
print('tau04 = ', tau04)
#
# taumax2k = (Tmax2*Tmax2)*const.Boltzmann/(W2k*H2)
# tau02k = taumax2*np.exp(-W2k/(const.Boltzmann*Tmax2k))
