import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp
from numpy import linspace, pi,exp, sin, cos, tan, loadtxt, savetxt, mean, zeros, size, log, sum, sqrt, arctan
from uncertainties import ufloat
import scipy.stats as st
from scipy.constants import c, e, h
from astropy.io import ascii
from scipy.integrate import quad
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 13
################
electricity1, temperature1, time1 = np.genfromtxt("measurement1.txt", unpack=True)
a_lin, a_error, b_lin, b_error = np.genfromtxt("measurement1_errors.txt", unpack=True)
#electricity1, temperature1, change_in_temperature1, time1 = np.genfromtxt("measurement2.txt", unpack=True)
#temperature1 *=10**(-11)
#################
#Aim 1 necessary activation energy W to chang the polarization
# by using equation (0) and (1)
# (0): y-axis ln(j), x-axis 1/T for one part of the measurement
# with j(T) = (p**2 * E)/(3*k*T_P) * (N_P)/(tau_0) * np.exp((-W)/(k*T))

# (1): look at notes
###############################
#try and error
###############################
runner = 0 # runner
total_i = 0
total_i_error = 0
i_T_total = 0
table = []
readings1 = temperature1 # in Kelvin
result1 = electricity1
def i(T,a,b,a_lin, b_lin):
    return T*a + b - T*a_lin + b_lin
def error(T, a_error, b_error):
    return T*a_lin + b_lin
def f(x, a, b):
    return a * x + b
while runner + 1 < len(readings1):
    a = (result1[runner+1]-result1[runner])/(readings1[runner+1]-readings1[runner])
    b = result1[runner]-a*readings1[runner]
    T1 = readings1[runner]
    T2 = readings1[runner+1]
    int_i = quad(i, T1, T2, args=(a, b, a_lin, b_lin))
    int_i_error = quad(error, T1, T2, args=(a_error, b_error))
    total_i = total_i + int_i[0]
    total_i_error = total_i_error + int_i_error[0]
    print("total_i",total_i)
    print("total_i_error",total_i_error)
    print("temperature",readings1[runner+1])
    table +=[readings1[runner+1]]
    table +=[total_i]
    table +=[total_i_error]
    table +=[result1[runner+1]]
    runner += 1
# print(table)
#np.savetxt('mes_1.txt', np.column_stack(table), fmt="%2.3f")
temperature_int, int_elec, int_elec_error, elec = np.genfromtxt("mes_1.txt", unpack=True)
errB = int_elec_error
plt.grid()
plt.errorbar(temperature_int, int_elec, xerr=0, yerr=errB, fmt='g-',label="Integral")
plt.legend(loc="best") #Positioniert automatisch die Legende
plt.xlabel(r'T / °C')
plt.ylabel(r' Integral i(T) / $10^{-11} \, \mathrm{A}$')
plt.savefig('test.pdf')  #zum Speichern der PDF-Datei
plt.show()
#print("runner",runner)
#(a)
y_achse_pol = np.log(int_elec/elec)
y_achse_pol_err = (1/(int_elec/elec) * 1/elec)**2 * int_elec_error**2
x_achse_pol = 1/(temperature_int)

def f(x, a, b):
    return a * x + b
#Definition der Ausgleichsfunktion
params, covariance = curve_fit(f, x_achse_pol, y_achse_pol)
#covariance is the covariance matrix
errors = np.sqrt(np.diag(covariance))

print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
print(y_achse_pol, "error",y_achse_pol_err, "x-achse",x_achse_pol)
x_plot = np.linspace(min(x_achse_pol),max(x_achse_pol))
plt.grid()
plt.errorbar(x_achse_pol, y_achse_pol, xerr=0, yerr=y_achse_pol_err, fmt='go',label="Messdaten")
plt.plot(x_plot, f(x_plot, *params), 'r-', label='Ausgleichsgerade' , linewidth=1)
plt.legend(loc="best") #Positioniert automatisch die Legende
plt.xlabel(r'1/T / 1/°C')
plt.ylabel(r'ln($\alpha$)')
plt.xlim(-0.5,0.5)
plt.savefig('int_1.pdf')  #zum Speichern der PDF-Datei
plt.show()
