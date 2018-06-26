import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat

########################################################################
##Kalibrierung
#############################################
print('Kalibrierung')
c = np.genfromtxt('Kalibrierung.txt')
n = 0
for i in range(0,511):
    if c[i] >2:
        n+=1

#print(n)
C = np.zeros((n-1,2))
#print(C)
j=1
for i in range(1,511):
    if c[i] >2:
        #print(C[j-1][0])
        if c[i-1] == 0:            #neuer Balken gefunden
            C[j][0] = c[i]
            C[j][1] = i
            j += 1
        else:                       #verschmierter Balken
            C[j-1][1] = (C[j-1][0]*C[j-1][1]+c[i]*i)/(C[j-1][0]+c[i])

#print(C)

c = C[1][1]
#print(c)
def f(x, a, b):
    return a*x+b

t = np.linspace (0,9,10)
for i in range(2, n-1):
    c =np.append(c,C[i][1]-C[i-1][1])

params, cov = curve_fit(f, t, C[:,1])
#plt.plot(f(t, *params), 'k-')
#plt.plot(t, C[:,1], 'gx')
#plt.grid()
#plt.show()

ct = params[0]

params = correlated_values(params, cov)
print('Fit(a, b): ')
for p in params:
    print(p)

m= np.mean(c)
sigma = np.std(c)/np.sqrt(len(c))

print('mean =', m, '+/-', sigma)
print('halfed =', m/2, '+-', sigma/2)


###########################################################
####### LEBENSDAUERBESTIMMUNG
########################################
print('Lebensdauerbestimmung: ')
counts = np.genfromtxt('lebensdauer.txt')

#print(c
#print(len(c))
#print(c)
C = np.array([counts[2]])
for i in range(3, 401):
    C = np.append(C,counts[i])
#print(C)
#print(len(C))
def e(t, a, b, d):
    return a*np.exp(-t/b)+d

t = np.linspace(0,len(C)-1,len(C))/ct
#print(t)

params , cov = curve_fit(e, t, C)

plt.plot(t,C, 'y.', label = 'Messwerte')
plt.plot(t, e(t, *params), 'b-', label = 'fit')

print('Fit: ')
params = correlated_values(params, cov)
for p in params:
    print(p)

plt.grid()
plt.legend(loc = 'best')
plt.xlabel(r'$t/\mu s$')
plt.ylabel('Counts')
#plt.yscale('log')
#plt.yscale('log')


plt.legend(loc='best')
plt.savefig('lebensdauer.pdf')
plt.clf()
#logcounts = np.log(counts)
