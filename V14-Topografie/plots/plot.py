import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)


##################################################################################
#################################################################################
####                        Quellenspektrum
####################################################################################
##################################################################################

N = np.genfromtxt('../Beste_Werte_wo_geht/Quellenspektrum.Spe', unpack = True)
c = np.arange(512)
N = N/30.116
max = 0
maxC = 0
for i in c:
    if(N[i]>max):
        max = N[i]
        maxC = i

c=c*661/maxC              #Umrechnung kanal keV
#Tabelle
# np.savetxt('tab.txt',np.column_stack([x,y]), delimiter=' & ',newline= r'\\'+'\n' )
#plt.subplot(1, 2, 1)
plt.plot([661,661],[0,200], 'b--', label ='Kante')
plt.errorbar(c, N,yerr=np.sqrt(N), fmt='g.', ecolor='r', label='Quellenspektrum')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
plt.xlabel('E/keV')
plt.ylabel('Ereignisse/100s')
plt.legend(loc='best')
plt.savefig('../build/Quellenspektrum.pdf')
plt.clf()
#plt.subplot(1, 2, 2)
#plt.plot(x, y, label='Kurve')
#plt.xlabel(r'$\alpha \:/\: \si{\ohm}$')
#plt.ylabel(r'$y \:/\: \si{\micro\joule}$')
#plt.legend(loc='best')

# in matplotlibrc leider (noch) nicht möglich
#plt.tight_layout(pad=0, h_pad=1.08, w_pad=1.08)
#plt.savefig('../build/plot2.pdf')
