import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat

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

params = correlated_values(params, cov)
for p in params:
    print(p)

m= np.mean(c)
print(m)
sigma = np.std(c)/np.sqrt(len(c))
print(sigma)
