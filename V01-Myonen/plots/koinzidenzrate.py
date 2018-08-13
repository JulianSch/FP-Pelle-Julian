import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
# from scipy.optimize import curve_fit
# import uncertainties.unumpy as unp
#############Werte auslesen &einpassen
l, r, c, t = np.genfromtxt('datenko.txt', unpack = True)
x = l-r
y = c/t

###########halbweetsbreiten tk gedöns
plateau = []
px = []

for i in range(0,len(y)-1):
    if y[i]>20:
        plateau.append(y[i])
        px.append(x[i])

plt.plot(px,plateau, 'b.', label = 'Plateauwerte')
plateau = np.mean(np.array(plateau))
print(plateau)
print('Halbwert: ', plateau/2)

def f(x,a,b):
    return a*x+b

xlinks = []
ylinks = []
xrechts = []
yrechts = []

for i in range(0,len(y)):
    if y[i] < 20 and y[i] > 2:
        if x[i]<0:
            ylinks.append(y[i])
            xlinks.append(x[i])
        if x[i] >0:
            yrechts.append(y[i])
            xrechts.append(x[i])

ylinks = np.array(ylinks)
xlinks = np.array(xlinks)
yrechts = np.array(yrechts)
xrechts = np.array(xrechts)

plt.grid()
plt.plot(x,y, 'rx', label ='Messwerte')
plt.plot(xlinks,ylinks,'k.', label = 'linke Flanke')
plt.plot(xrechts,yrechts,'g.', label = 'rechte Flanke')
plt.xlabel(r'Verzögerungszeit in ns')
plt.ylabel(r'Zählrate')


pl,cl = curve_fit(f,xlinks,ylinks)
pr,cr = curve_fit(f,xrechts,yrechts)

Pl = correlated_values(pl, cl)
print('links Fit(a, b): ')
for p in Pl:
    print(p)

Pr = correlated_values(pr, cr)
print('rechts Fit(a, b): ')
for p in Pr:
    print(p)


xl = (plateau/2-Pl[1])/Pl[0]
xr = (plateau/2-Pr[1])/Pr[0]
print(xl,xr)
print('tk = ', 40 - (xr-xl))


plt.plot([-30,40],[plateau,plateau],'b-', label = 'Plateaumittel')
plt.plot([-30,40],[plateau/2,plateau/2], 'k-.', label = 'Halbe Höhe')
plt.plot(xlinks,f(xlinks,*pl), 'r--', label = 'Flankenausgleich')
plt.plot(xrechts,f(xrechts,*pr), 'r--')
plt.legend(loc='best')
plt.savefig('plot.pdf')
