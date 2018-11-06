import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
# import scipy.stats as stats
from scipy.optimize import curve_fit
from uncertainties import correlated_values  # , correlation_matrix, ufloat
import uncertainties.unumpy as unp
from uncertainties.unumpy import (nominal_values as noms)  # ,std_devs as stds)


def riemann(temp, strom, b, e):
    a = 0
    a1 = 0
    for i in range(b, e, 1):
        t = (temp[i] - temp[i+1])
        s = (strom[i] + strom[i+1])/2
        a += np.sqrt((s*t)**2)
        a1 += s*t
    # print('da=', a-a1)
    return a


def depol(temp, W, A, a):
    return a+A*np.exp(-W/(temp))

T1, I1 = np.genfromtxt('daten2.txt', unpack=True)
T2, I2 = np.genfromtxt('daten3.txt', unpack=True)
t1, i1 = np.genfromtxt('ignoredaten2.txt', unpack=True)
t2, i2 = np.genfromtxt('ignoredaten3.txt', unpack=True)
I1 *= -1
I2 *= -1
I1 -= -0.12
I2 -= -0.15
T1 += 273.15
T2 += 273.15
i1 -= -0.12
i2 -= -0.15
t1 = t1+273.15
i1 = -i1
t2 = t2+273.15
i2 = -i2

params1, cov1 = curve_fit(depol, t1, i1)
Params1 = correlated_values(params1, cov1)
print(Params1)

y1 = depol(T1, *noms(Params1))
f1 = I1-y1


params2, cov2 = curve_fit(depol, t2, i2)
print(params2)
Params2 = correlated_values(params2, cov2)

y2 = depol(T2, *noms(Params2))
f2 = I2-y2

###############################################################################
# Integral
#############################################################################
print('T1/I1 size: ', T1.size, f1.size)
print('T2/I2 size: ', T2.size, f2.size)
int1 = []
int2 = []
for i in range(0, 41):
    int1.append(riemann(T1, f1, i, 12))
for i in range(0, 108):
    int2.append(riemann(T2, f2, i, 49))


# print('int(I1)= ',int1)
# print('int(I2)= ',int2)

print(len(int1))
print(len(int2))
b1 = 3.2/60   # Kelvin*sekunde^-1
b2 = 1.26/60

print('b1= ', b1)
print('b2 = ', b2)
tau1 = int1/(b1*f1)
tau2 = int2/(b2*f2)
copy1 = tau1
copy2 = tau2

plt.xlabel('T/K')
plt.ylabel(r'$\tau$/s')
plt.grid()
plt.legend(loc='best')
plt.plot(T1, tau1, 'r.', label='Erste Messreihe')
plt.savefig('tau.pdf')
plt.clf()

# t1 = T1[0:12]
# t2 = T2[18:49]
while len(copy1) < len(copy2):
    copy1 = np.append(copy1, 0)
    t1 = np.append(t1, 0)
print(len(copy1), len(copy2), len(t1), len(t2))
# np.savetxt('tautab.txt', np.column_stack([copy1, t1, copy2, t2]), delimiter=' & ', newline=r'\\'+'\n')

print('taumax1= ', tau1[7])

plt.plot(T2, tau2, 'g.', label='Zweite Messreihe')
plt.xlabel('T/K')
plt.ylabel(r'$\tau$/s')
plt.grid()
plt.legend(loc='best')
plt.savefig('tau2.pdf')
plt.clf()

Tau1 = np.log(int1/f1)
Tau2 = np.log(int2/f2)
# t1 = 1/T1[0:12]
# t2 = 1/T2[18:49]


def f(x, m, b):
    return m*x+b
plarams1, cov1 = curve_fit(f, t1, Tau1)
Plarams1 = correlated_values(plarams1, cov1)

print('m,b = ', Plarams1)
k = unp.ufloat(const.k, 0)
print('kb=', k)

w1 = k*Plarams1[0]

print('W1=', w1)

plt.plot(t1, Tau1, 'r.', label='Erste Messreihe')
plt.xlabel('1/T in 1/K')
plt.ylabel(r'$\ln(\tau \H)}$')
plt.grid()
plt.plot(t1, f(t1, *plarams1), 'r--', label='Erste Ausgleichsgerade')
plt.savefig('integral.pdf')
plt.clf()

plarams2, cov2 = curve_fit(f, t2, Tau2)
Plarams2 = correlated_values(plarams2, cov2)

print('m,b = ', Plarams2)
w2 = k*Plarams2[0]
print('W2=', w2)

plt.plot(t2, Tau2, 'g.', label='Zweite Messreihe')
plt.plot(t2, f(t2, *plarams2), 'g--', label='Zweite Ausgleichsgerade')
plt.xlabel('1/(T/K)')
plt.ylabel(r'$\ln(\tau \H)}$')
plt.grid()
plt.legend(loc='best')
plt.savefig('integral2.pdf')
plt.clf()

Tmax1 = T1[7]
Tmax2 = T2[32]
print('Tmax1= ', Tmax1)
print('Tmax2= ', Tmax2)

taumax1 = (Tmax1*Tmax1)*k/w1/b1
taumax2 = (Tmax2*Tmax2)*k/w2/b2

print('taumax1= ', taumax1)
print('taumax2= ', taumax2)

tau01 = taumax1*unp.exp(-w1/(k*Tmax1))
tau02 = taumax2*unp.exp(-w2/(k*Tmax2))

print('tau01= ', tau01)
print('tau02= ', tau02)

# Int1 = riemann(T1,I1,1, T1.size)
# Int2 = riemann(T2,I2,1, T2.size)
# print('Int(I1)= ',Int1)
# print('Int(I2)= ',Int2)
