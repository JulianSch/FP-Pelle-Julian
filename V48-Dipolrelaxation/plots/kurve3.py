import matplotlib.pyplot as plt
import numpy as np
# import scipy.constants as const
import scipy.stats as stats
# from scipy.optimize import curve_fit
# from uncertainties import correlated_values, correlation_matrix
T1, I1 = np.genfromtxt('daten2.txt', unpack=True)
T2, I2 = np.genfromtxt('daten3.txt', unpack=True)
I1-=-0.12
I2-=-0.15

T1 = T1+273.15
I1 = -I1
diff = np.array([])
for i in range(len(T1)-1):  # geht länge des arrays durch
    d = T1[i]-T1[i+1]
    diff = np.append(diff, d)

print(diff)

print(np.mean(diff))
print(stats.sem(diff))

plt.plot(T1, I1, 'r.', label='Erste Messreihe')

# plt.savefig('kurve2.pdf')

T2 = T2+273.15
I2 = -I2
diff = np.array([])
for i in range(len(T2)-1):  # geht länge des arrays durch
    d = T2[i]-T2[i+1]
    diff = np.append(diff, d)

print(diff)

print(np.mean(diff))
print(stats.sem(diff))


plt.plot(T2, I2, 'g.', label='Zweite Messreihe')
plt.xlabel('T in K')
plt.ylabel('I in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('kurve3.pdf')
