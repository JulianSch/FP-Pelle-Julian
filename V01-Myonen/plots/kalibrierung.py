import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp

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
        print(C[j-1][0])
        if c[i-1] == 0:            #neuer Balken gefunden
            C[j][0] = c[i]
            C[j][1] = i
            j += 1
        else:                       #verschmierter Balken
            C[j-1][1] = (C[j-1][0]*C[j-1][1]+c[i]*i)/(C[j-1][0]+c[i])


print(C)

c = C[1][1]
print(c)

for i in range(2,n-1):
    c =np.append(c,C[i][1]-C[i-1][1])
print(c)

m= np.mean(c)
print(m)
sigma = np.std(c)/np.sqrt(len(c))
print(sigma)
