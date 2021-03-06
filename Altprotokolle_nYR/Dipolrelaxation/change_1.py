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
########################
#HIER ANPASSEN
########################
electricity_0, temperature, time, rate = np.genfromtxt("measurement.txt", unpack=True)
electricity = rate * electricity_0 / 10
i=0
jj = 0
log_j = 0
temperature = temperature +273.15 #in Kelvin
log_j_save = []
log_j_aus = []
tem_j =[]
tem_j_aus =[]
r = 1.5 * 10**(-3)
def f(x, a, b):
    return a * x + b

while jj + 1 < len(electricity):
    ###############
    if electricity[jj] > 0:
        log_j = np.log(electricity*10**(-11)/(r**2))
        log_j_save = np.append(log_j_save, log_j[jj])
        tem_j = np.append(tem_j, temperature[jj])
    if 1/temperature[jj] > 0.003870  and 1/temperature[jj] < 0.00427 :
        log_j_aus = np.append(log_j_aus, log_j[jj])
        tem_j_aus = np.append(tem_j_aus, temperature[jj])
    jj += 1
np.savetxt('measurement1_j.txt', np.column_stack([log_j_save, tem_j]), fmt="%2.3f", delimiter=" ", header='log_j,tem')
np.savetxt('measurement1_j_tab.txt', np.column_stack([log_j_save, tem_j]), fmt="%2.3f", delimiter=" & ", header='log_j,tem')
##########
#Definition der Ausgleichsfunktion
params_aus, covariance_aus = curve_fit(f, 1/tem_j_aus, log_j_aus)
#covariance is the covariance matrix
errors_aus = np.sqrt(np.diag(covariance_aus))

print('a_j =', params_aus[0], '±', errors_aus[0])
print('b_j =', params_aus[1], '±', errors_aus[1])
x_plot_aus = np.linspace(min(1/tem_j_aus),max(1/tem_j_aus))
plt.grid()
plt.plot(1/tem_j, log_j_save, 'ro',label="Messpunkte", linewidth=1)
plt.plot(x_plot_aus, f(x_plot_aus, *params_aus), 'g--', label='lineare Steigung' , linewidth=2)
plt.legend(loc="best") #Positioniert automatisch die Legende
plt.xlabel(r'1/T / 1/K')
plt.ylabel(r' ln(j)')
########################
#HIER ANPASSEN
########################
plt.savefig('Messdaten_1_lnj.pdf')  #zum Speichern der PDF-Datei
plt.show()
############################## heizrate
#Definition der Ausgleichsfunktion
params_rate, covariance_rate = curve_fit(f, time, temperature)
#covariance is the covariance matrix
errors_rate = np.sqrt(np.diag(covariance_rate))

print('a_rate =', params_rate[0], '±', errors_rate[0])
print('b_rate =', params_rate[1], '±', errors_rate[1])
x_plot_rate = np.linspace(min(time),max(time))
plt.grid()
plt.plot(time, temperature, 'bo',label="Messpunkte", linewidth=1)
plt.plot(x_plot_rate, f(x_plot_rate, *params_rate), 'r--', label='Ausgleichsgerade' , linewidth=3)
plt.legend(loc="best") #Positioniert automatisch die Legende
plt.ylabel(r'T / K')
plt.xlabel(r't / s')
########################
#HIER ANPASSEN
########################
plt.savefig('Messdaten_1_rate.pdf')  #zum Speichern der PDF-Datei
plt.show()
##############################

temperature_xl = []
electricity_yl = []
electricity_y = []
temperature_x = []

########################
#HIER ANPASSEN
########################
while i < len(temperature):
    if temperature[i] > (220.15) and temperature[i] < (232.15) or temperature[i] > 277 and temperature[i] < 289.5:
        temperature_xl = np.append(temperature_xl, temperature[i])
        electricity_yl = np.append(electricity_yl, electricity[i])
    else:
        temperature_x = np.append(temperature_x, temperature[i])
        electricity_y = np.append(electricity_y, electricity[i])
    i += 1
#Definition der Ausgleichsfunktion
params, covariance = curve_fit(f, temperature_xl, electricity_yl)
#covariance is the covariance matrix
errors = np.sqrt(np.diag(covariance))

print('a =', params[0], '±', errors[0])
print('b =', params[1], '±', errors[1])
######################################
temperature_xst = []
electricity_yst = []
j = 0
while j < len(temperature):
    if temperature[j] > 233.15 and temperature[j] < 258.15:
        temperature_xst = np.append(temperature_xst, temperature[j])
        electricity_yst = np.append(electricity_yst, electricity[j])
    j += 1
#Definition der Ausgleichsfunktion
params1, covariance1 = curve_fit(f, temperature_xst, electricity_yst)
#covariance is the covariance matrix
errors1 = np.sqrt(np.diag(covariance1))

print('a_begin =', params1[0], '±', errors1[0])
print('b_begin =', params1[1], '±', errors1[1])
x_plot_1 = np.linspace(temperature_xst[0],temperature_xst[len(temperature_xst)-1])
x_plot = np.linspace(temperature_xl[0],temperature_xl[len(temperature_xl)-1])
plt.grid()
plt.plot(temperature_xl, electricity_yl, 'ro',label="Messpunkte (Untergrund)", linewidth=1)
plt.plot(temperature_x, electricity_y, 'bo',label="Messpunkte", linewidth=1)
plt.plot(x_plot, f(x_plot, *params), 'r-', label='Ausgleichsgerade' , linewidth=1)
#plt.plot(x_plot_1, f(x_plot_1, *params1), 'g--', label='lineare Steigung' , linewidth=1)
plt.legend(loc="best") #Positioniert automatisch die Legende
plt.xlabel(r'T / K')
plt.ylabel(r' i(T) / $10^{-11} \, \mathrm{A}$')
########################
#HIER ANPASSEN
########################
plt.savefig('Messdaten_1.pdf')  #zum Speichern der PDF-Datei
plt.show()
########################
#HIER ANPASSEN UND AUSKOMMENTIEREN
########################
np.savetxt('measurement1.txt', np.column_stack([electricity, temperature, time]), fmt="%2.3f", delimiter=" ", header='electricity, temperature, time')
np.savetxt('measurement1_tab.txt', np.column_stack([electricity, temperature, time]), fmt="%2.3f", delimiter=" & ", header='electricity, temperature, time')
np.savetxt('measurement1_errors.txt', np.column_stack([params[0], errors[0], params[1], errors[1]]), header='params[0], errors[0], params[1], errors[1]')
