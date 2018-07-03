import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.stats as stats
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix, ufloat


def riemann(temp, strom, b, e):
    a = 0
    a1 = 0
    for i in range(b, e, 1):
        t = (temp[i] - temp[i+1])
        s = (strom[i] + strom[i+1])/2
        a += np.sqrt((s*t)**2)
        a1 += s*t
    #print('da=', a-a1)
    return a

def depol(temp, W, A,a):
    return a+A*np.exp(-W/(temp))

T1, I1 = np.genfromtxt('daten2.txt', unpack=True)
T2, I2 = np.genfromtxt('daten3.txt', unpack=True)
I1*=-1
I2*=-1
I1-=-0.12
I2-=-0.15
T1+=273.15
T2+=273.15

####################################################
###############################################
############ Untergrund fit
##########################
##############################
#params1, cov1 = curve_fit(depol, T1[13:34], I1[13:34])
#print(params1)
#Params1 = correlated_values(params1, cov1)
#params2, cov2 = curve_fit(depol, T2[49:88], I2[49:88])
#Params2 = correlated_values(params2, cov2)
plt.plot(T1[13:34],I1[13:34],'r.', label= 'Erste Messreihe')
plt.plot(T2[49:88],I2[49:88], 'g.', label = 'Zweite Messreihe')
#plt.plot(T1[13:34], depol(T1[13:34],*params1), 'r-', label = 'Erster Fit')
#plt.plot(T2[49:88], depol(T2[49:88],*params2), 'g-', label = 'Zweiter Fit')

plt.xlabel('T in K')
plt.ylabel('I in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('untergrund.pdf')
plt.clf()

################################################################################
############################################################################
##### Integral wichse
######################################################################
#############################################################################
print('T1/I1 size: ', T1.size, I1.size)
print('T2/I2 size: ', T2.size, I2.size)
int1=[]
int2= []
for i in range(0,12):
    int1.append(riemann(T1, I1,i,12))
for i in range(18,49):
    int2.append(riemann(T2, I2,i,49))


#print('int(I1)= ',int1)
#print('int(I2)= ',int2)

print(len(int1))
print(len(int2))
b1 = 3.2
b2 = 1.26
tau1 = int1/(b1*I1[1:13])
tau2 = int2/(b2*I2[18:49])

plt.plot(T1[1:13],tau1, 'r.', label='Erste Messreihe')
plt.plot( T2[18:49], tau2, 'g.', label='Zweite Messreihe')

plt.xlabel('T/K')
plt.ylabel(r'$\tau$/ps')
plt.grid()
plt.legend(loc='best')
plt.savefig('integral.pdf')
plt.clf()

Tau1 = np.log(tau1*b1)
Tau2 = np.log(tau2*b1)
t1 = 1/T1[1:13]
t2 = 1/T2[18:49]

def f(x,m,b):
    return m*x+b
params1,cov1 = curve_fit(f,t1,Tau1)
Params1 = correlated_values(params1, cov1)
print('m,b = ',Params1)
k = ufloat(const.k,0)
w1 = k*Params1[0]
print('W1=', w1)
plt.plot(t1, Tau1, 'r.', label ='Erste Messreihe')
plt.plot(t1, f(t1,*params1), 'r--', label ='Erster Fit')

params2,cov2 = curve_fit(f,t2,Tau2)
Params2 = correlated_values(params2, cov2)
print('m,b = ',Params2)
k = ufloat(const.k,0)
w2 = k*Params2[0]
print('W2=', w2)
plt.plot(t2, Tau2, 'g.', label ='Erste Messreihe')
plt.plot(t2, f(t2,*params2), 'g--', label ='Erster Fit')
plt.xlabel('1/(T/K)')
plt.ylabel(r'$\ln(\tau\cdot H)$')
plt.grid()
plt.legend(loc='best')
plt.savefig('log.pdf')

#Int1 = riemann(T1,I1,1, T1.size)
#Int2 = riemann(T2,I2,1, T2.size)
#print('Int(I1)= ',Int1)
#print('Int(I2)= ',Int2)
