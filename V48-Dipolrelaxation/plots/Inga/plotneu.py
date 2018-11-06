import numpy as np
#from textable import table
from scipy import constants
import scipy.integrate as integrate
from uncertainties import ufloat
import matplotlib.pyplot as plt
import uncertainties
import uncertainties.unumpy as unp
from scipy.optimize import curve_fit
# from scipy.integrate import quad, trapz
import sympy


def untersumme(temp, strom):
    a = 0
    if (temp.size == 18):
        for i in range(0, 17, 1):
            t = (temp[i] - temp[i+1])
            s = (strom[i] + strom[i+1])/2
            a += np.sqrt((s*t)**2)
        print('Modus 0')
        return a
    elif(temp.size == 35):
        for i in range(0, 34, 1):
            t = (temp[i] - temp[i+1])
            s = (strom[i] + strom[i+1])/2
            a += np.sqrt((s*t)**2)
        print('Modus 1')
        return a
    else:
        print('Nix gemacht')
        return a

temp1, strom1 = np.genfromtxt('daten2.txt', unpack=True)
# s,
temp2, strom2 = np.genfromtxt('daten3.txt', unpack=True)

temp1 += 273.15
temp2 += 273.15
strom1 *= 10**(-15)
strom2 *= 10**(-14)  # war anders abgespeichert
t_err = 0.05
s_err = 0.05*10**(-15)

# heizi1 = []
# heizi2 = []

# for i in range(0, 47, 1):
#    heizi1.append(temp1[i] - temp1[i+1])

# for i in range(0, 78, 1):
    # heizi2.append(temp2[i] - temp2[i+1])

# heiz1 = ufloat(np.mean(heizi1), np.std(heizi1))
# heiz2 = ufloat(np.mean(heizi2), np.std(heizi2))

#
def func(x, a, b):
     return b * np.exp(a/x)
#
#
t1 = temp1[0:9]
s1 = strom1[0:9]
t2 = temp2[9:30]
s2 = strom2[9:30]
#
# params1, covariance1 = curve_fit(func, t1, (s1))
# errors1 = np.sqrt(np.diag(covariance1))
# print('a1 =', params1[0], '±', errors1[0])
# print('b1 =', params1[1], '±', errors1[1])
# a1 = ufloat(params1[0], errors1[0])
# b1 = ufloat(params1[1], errors1[1])
#
# params2, covariance2 = curve_fit(func, t2, (s2))
# errors2 = np.sqrt(np.diag(covariance2))
# print('a2 =', params2[0], '±', errors2[0])
# print('b2 =', params2[1], '±', errors2[1])
# a2 = ufloat(params2[0], errors2[0])
# b2 = ufloat(params2[1], errors2[1])
#
# plt.clf()
# x = np.linspace(241, 261, 100)
#
# plt.plot(x, params1[1]*np.exp(params1[0]/x), 'g-', label='Ersten Messreihe - Fit', linewidth=2)
# plt.plot(temp1, strom1, 'gx', label='Ersten Messreihe ', linewidth=2)
# plt.plot(t1, s1, 'rx', label='Ersten Messreihe - Messwerte - Messwerte für den Fit ', linewidth=2)
# plt.xlabel(r'$T \:/\: K$')
# plt.ylabel(r'$ln(j)$')
# plt.yscale("log")
# plt.grid()
# plt.errorbar(t1, s1, xerr=t_err, yerr=s_err, ecolor='k', fmt='None')
# plt.legend(loc="best")
# # plt.tight_layout()
# plt.savefig('plot1.pdf')
# plt.clf()
#
# x = np.linspace(225, 265, 500)
# plt.plot(x, params2[1]*np.exp(params2[0]/x), 'b-', label='Zweite Messreihe - Fit', linewidth=2)
# plt.plot(temp2, strom2, 'rx', label='Zweite Messreihe - Messwerte', linewidth=2)
# plt.plot(t2, s2, 'bx', label='Zweite Messreihe - Messwerte - für den Fit', linewidth=2)
#
# plt.xlabel(r'$ T \:/\: K$')
# plt.ylabel(r'$ln(j)$')
# plt.yscale("log")
# plt.grid()
# plt.legend(loc="best")
# plt.errorbar(t2, s2, xerr=t_err, yerr=s_err, ecolor='k', fmt='None')
# plt.savefig('plot2.pdf')
# plt.clf()
# print("--------------------------")
#
# print('heizrate1 ', '{:L}'.format(heiz1))
# print('heizrate2 ', '{:L}'.format(heiz2))
#
# print("--------------------------")
#
# W1 = - constants.k*a1
# W2 = - constants.k*a2
# print("W1 in eV beträgt", '{:L}'.format(W1/constants.e))
# print("W2 in eV beträgt", '{:L}'.format(W2/constants.e))
# print("--------------------------")
Tmax1 = ufloat(-12.5+273.15, 0.05)
Tmax2 = ufloat(-13.1+273.15, 0.05)
# print('Tau(Tmax) für die 1 Messung beträgt', '{:L}'.format(heiz1/60 * a1/Tmax1))
# print('Tau(Tmax) für die 2 Messung beträgt', '{:L}'.format(heiz2/60 * a2/Tmax2))
# print("--------------------------")
# t01 = ((heiz1*a1)/(60*Tmax1))*unp.exp(-a1/Tmax1)
# t02 = ((heiz2*a2)/(60*Tmax2))*unp.exp(-a2/Tmax2)
# print('Tau_0 für die 1 Messung beträgt', t01, '{:L}'.format(t01))
# print('Tau_0 für die 2 Messung beträgt', t02, '{:L}'.format(t02))
# # Tabellen
# strom1 *= 10**(15)
# strom2 *= 10**(15)
# np.savetxt('MessungWir.txt', np.column_stack([temp1, strom1]),
#            delimiter=' & ', newline=r' \\'+'\n', fmt="%.2f")
# np.savetxt('MessungEr.txt', np.column_stack([temp2, strom2]),
#            delimiter=' & ', newline=r' \\'+'\n', fmt="%.2f")
# # nun die zweite Methode
# print("Zweite Methode: Integration")
tt1 = temp1[0:18]
ss1 = strom1[0:18]
tt2 = temp2[5:40]
ss2 = strom2[5:40]
# fitte erneut den strom mit der Temperatur und berechne anschließend das integral
def fitti(T, C1, W):
        return (C1 * np.exp(- W / (constants.k * T)))


def integriere(C1, A, tmin, tmax):
        y1, err1 = integrate.quad(lambda c, a, x: c*unp.exp(a/x), tmin, tmax, args=(C1[0]+C1[1],A[0]+A[1]))
        y2, err2 = integrate.quad(lambda c, a, x: c*unp.exp(a/x), tmin, tmax, args=(C1[0]+C1[1],A[0]))
        y3, err3 = integrate.quad(lambda c, a, x: c*unp.exp(a/x), tmin, tmax, args=(C1[0],A[0]+A[1]))
        y4, err4 = integrate.quad(lambda c, a, x: c*unp.exp(a/x), tmin, tmax, args=(C1[0],A[0]))
        y5, err5 = integrate.quad(lambda c, a, x: c*unp.exp(a/x), tmin, tmax, args=(C1[0]-C1[1], A[0]-A[1]))
        y6, err6 = integrate.quad(lambda c, a, x: c*unp.exp(a/x), tmin, tmax, args=(C1[0]-C1[1], A[0]))
        y7, err7 = integrate.quad(lambda c, a, x: c*unp.exp(a/x), tmin, tmax, args=(C1[0],A[0]-A[1]))
        print('Y1,Err1 ', '{:L}'.format(ufloat(y1, err1)))
        print('Y2,Err2 ', '{:L}'.format(ufloat(y2, err2)))
        print('Y3,Err3 ', '{:L}'.format(ufloat(y3, err3)))
        print('Y4,Err4 ', '{:L}'.format(ufloat(y4, err4)))
        print('Y5,Err5 ', '{:L}'.format(ufloat(y5, err5)))
        print('Y6,Err6 ', '{:L}'.format(ufloat(y6, err6)))
        print('Y7,Err7 ', '{:L}'.format(ufloat(y7, err7)))
        return 0

params, covariance = curve_fit(func, tt1, ss1)
errors = np.sqrt(np.diag(covariance))
a1 = ufloat(params[0], errors[0])  # *constants.k*(-1)
a1n = [params[0], errors[0]]
C1n = [params[1], errors[1]]
C11 = ufloat(params[1], errors[1])
plt.clf()
print('{:L}'.format(ufloat(params[0], errors[0])), "Ergebnisse func Parameter Nr1 Messung 2")
print('{:L}'.format(ufloat(params[1], errors[1])), "Ergebnisse func Parameter Nr2 Messung 2")
plt.plot(tt1, ss1, 'g.', label='Daten \n für die Integration verwendet')
plt.plot(temp1[18:48], strom1[18:48], 'b.', label='Restliche Daten ')
# plt.plot(x, params[1]*np.exp(params[0]/x) , 'b-', label='Erste Messreihe - Fit der e-Funktion',linewidth=2)
plt.xlabel(r'$T$ / K')
plt.ylabel(r'$I$ / pA')
plt.errorbar(tt1, ss1, xerr=t_err, yerr=s_err, ecolor='k', fmt='None')
plt.legend(loc='best')
plt.savefig('plot11.pdf')
plt.clf()


params, covariance = curve_fit(func, tt2, ss2)
errors = np.sqrt(np.diag(covariance))
a2 = ufloat(params[0], errors[0])  # *constants.k*(-1)
C12 = ufloat(params[1], errors[1])
print('{:L}'.format(ufloat(params[0], errors[0])), "Ergebnisse func Parameter Nr1 Messung 2")
print('{:L}'.format(ufloat(params[1], errors[1])), "Ergebnisse func Parameter Nr2 Messung 2")
a2n = [params[0], errors[0]]
C2n = [params[1], errors[1]]

plt.plot(tt2, ss2, 'g.', label='Daten \n für die Integration verwendet')
plt.plot(temp2[40:78], strom2[40:78], 'b.', label='Restliche Daten ')
# plt.plot(x, params[1]*np.exp(params[0]/x) , 'b-', label='Zweite Messreihe - Fit der e-Funktion',linewidth=2)
plt.errorbar(tt2, ss2, xerr=t_err, yerr=s_err, ecolor='k', fmt='None')
plt.xlabel(r'$T$ / K')
plt.ylabel(r'$I$ / pA')
plt.legend(loc='best')
plt.savefig('plot12.pdf')
plt.clf()
# assi= integriere(C1n, a1n,tt1[0],tt1[16])
# assi= integriere(C2n, a2n,tt2[0],tt2[34])
# print('Ergebnisse der Untersumme für die Messreihe1', untersumme(tt1,ss1))
# print('Ergebnisse der Untersumme für die Messreihe2', untersumme(tt2,ss2))
# mit integral:
# int1= ufloat(6081.089781289159 ,  6.751365889999982e-11)
# int2= ufloat(108.30309786312974, 1.2024059288587822e-12)
# mit untersumme:
int1 = ufloat(275.9125, 0.1)
int2 = ufloat(323.825, 0.1)


def fit(inteedurchstrom, k, z):
    return (np.log(inteedurchstrom) - k) * z


y1 = uncertainties.nominal_value(int1) / ss1
t1 = 1/tt1
y = np.array([sympy.N(i) for i in y1], dtype='float64')
t = np.array([sympy.N(i) for i in t1], dtype='float64')

params, covariance = curve_fit(fit, y, t)
errors = np.sqrt(np.diag(covariance))
print('{:L}'.format(ufloat(params[0], errors[0])), "Ergebnisse fit Parameter Nr1 Messung 1")
print('{:L}'.format(ufloat(params[1], errors[1])), "Ergebnisse fit Parameter Nr2 Messung 1")
# print(constants.k /( ufloat(params[1],errors[1]) )  , ' W1 ')
print('W1 in eV ', '{:L}'.format(constants.k / (ufloat(params[1], errors[1])) / constants.e))


y2 = uncertainties.nominal_value(int2) / ss2
t2 = 1/tt2
y = np.array([sympy.N(i) for i in y2], dtype='float64')
t = np.array([sympy.N(i) for i in t2], dtype='float64')
# print('y' , y , ' t' , t , ' Fit 2 ' )
params, covariance = curve_fit(fit, y, t)
errors = np.sqrt(np.diag(covariance))
print('{:L}'.format(ufloat(params[0], errors[0])), "Ergebnisse fit Parameter Nr1 Messung 2")
print('{:L}'.format(ufloat(params[1], errors[1])), "Ergebnisse fit Parameter Nr2 Messung 2")
# b=ufloat(params,errors)
# print(constants.k /( ufloat(params[1],errors[1]) )  , ' W2 ')
print('W2 in eV ', constants.k / (ufloat(params[1],errors[1]) * constants.e ))
print('{:L}'.format(constants.k / (ufloat(params[1],errors[1]) * constants.e)))

print('Methode 2 Tau(Tmax) für die 1 Messung beträgt', '{:L}'.format(heiz1/60 * a1/Tmax1))
print('Methode 2 Tau(Tmax) für die 2 Messung beträgt', '{:L}'.format(heiz2/60 * a2/Tmax2))

print("--------------------------")
t01 = ((heiz1*a1)/(60*Tmax1))*unp.exp(-a1/Tmax1)
t02 = ((heiz2*a2)/(60*Tmax2))*unp.exp(-a2/Tmax2)
print('Methode 2 Tau_0 für die 1 Messung beträgt', '{:L}'.format(t01))
print('Methode 2 Tau_0 für die 2 Messung beträgt', '{:L}'.format(t02))
