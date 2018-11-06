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
electricity1, temperature1, time1 = np.genfromtxt("measurement2.txt", unpack=True)
a_lin, a_error, b_lin, b_error = np.genfromtxt("measurement2_errors.txt", unpack=True)
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
readings_save = []
total_i_save =[]
total_i_error_save =[]
result1_save =[]
readings1 = temperature1 # in Kelvin
result1 = electricity1
def i(T,a,b,a_lin, b_lin):
    return T*a + b - T*a_lin - b_lin
def error(T, a_error, b_error):
    return np.sqrt((T**2/2)**2 * a_error**2 + T**2 * b_error**2)
def f(x, a, b):
    return a * x + b
while runner + 1 < len(readings1):
    if readings1[len(readings1)-(1+runner)] < 278 and readings1[len(readings1)-(1+runner)] > 225:
        a = (result1[len(readings1)-(runner+1)]-result1[len(readings1)-(runner+2)])/(readings1[len(readings1)-(runner+1)]-readings1[len(readings1)-(runner+2)])
        b = result1[len(readings1)-(runner+1)]-a*readings1[len(readings1)-(runner+1)]
        print(a)
        T1 = readings1[len(readings1)-(runner+2)]
        T2 = readings1[len(readings1)-(runner+1)]
        int_i = quad(i, T1, T2, args=(a, b, a_lin, b_lin))
        int_i_error = error(T2-T1, a_error, b_error)
        #int_i_error = quad(error, T1, T2, args=(a_error, b_error))
        total_i = total_i + int_i[0]
        total_i_error = total_i_error + int_i_error
        print("total_i",total_i)
        print("total_i_error",total_i_error)
        print("temperature",readings1[len(readings1)-(1+runner)])
        readings_save = np.append(readings_save, readings1[len(readings1)-(runner+2)])
        total_i_save = np.append(total_i_save, total_i)
        total_i_error_save =np.append(total_i_error_save, total_i_error)
        result1_save = np.append(result1_save,result1[len(readings1)-(runner+2)])
    runner += 1
#print(table)
np.savetxt('mes_2.txt', np.column_stack([readings_save, total_i_save,total_i_error_save,result1_save]), fmt="%2.3f", header='readings_save, total_i_save,total_i_error_save,result1_save')
np.savetxt('mes_2_tab.txt', np.column_stack([readings_save, total_i_save,total_i_error_save,result1_save]), fmt="%2.3f", delimiter=" & ", header='readings_save, total_i_save,total_i_error_save,result1_save')

temperature_int, int_elec, int_elec_error, elec = np.genfromtxt("mes_2.txt", unpack=True)
errB = int_elec_error
plt.grid()
plt.errorbar(temperature_int, int_elec, xerr=0, yerr=errB, fmt='g-',label="Integral")
plt.legend(loc="best") #Positioniert automatisch die Legende
plt.xlabel(r'T / °C')
plt.ylabel(r' Integral i(T) / $10^{-11} \, \mathrm{A}$')
plt.savefig('test2.pdf')  #zum Speichern der PDF-Datei
plt.show()
#print("runner",runner)
#(a)
y_achse_pol = np.log(int_elec/elec)
y_achse_pol_err =  np.sqrt((1/int_elec)**2 * int_elec_error**2)
x_achse_pol = 1/(temperature_int)

print(y_achse_pol_err)
x_achse_xst = []
y_achse_yst = []
j = 0
while j < len(x_achse_pol):
    if x_achse_pol[j] > 0.0037 and x_achse_pol[j] < 0.0041:
        x_achse_xst = np.append(x_achse_xst, x_achse_pol[j])
        y_achse_yst = np.append(y_achse_yst, y_achse_pol[j])
        j += 1
    else:
        j += 1
#Definition der Ausgleichsfunktion
params1, covariance1 = curve_fit(f, x_achse_xst, y_achse_yst)
#covariance is the covariance matrix
errors1 = np.sqrt(np.diag(covariance1))

print('a_begin =', params1[0], '±', errors1[0])
print('b_begin =', params1[1], '±', errors1[1])
plt.grid()
#plt.yscale('log')
x_plot = np.linspace(min(x_achse_pol),max(x_achse_pol))
plt.errorbar(x_achse_pol, y_achse_pol, xerr=0, yerr=y_achse_pol_err, fmt='go',label="Messdaten")
plt.plot(x_plot, f(x_plot, *params1), 'r-', label='Ausgleichsgerade' , linewidth=1)
plt.legend(loc="best") #Positioniert automatisch die Legende
plt.xlabel(r'1/T / 1/K')
plt.ylabel(r"$\ln{\left( \frac{i_\mathrm{Int}}{i(T_2)}\right)} $")
#plt.xlim(-0.5,0.5)
plt.savefig('int_2.pdf')  #zum Speichern der PDF-Datei
plt.show()
