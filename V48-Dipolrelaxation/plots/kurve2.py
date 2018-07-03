import matplotlib.pyplot as plt
import numpy as np
# import scipy.constants as const
import scipy.stats as stats
# from scipy.optimize import curve_fit
# from uncertainties import correlated_values, correlation_matrix
T1, I1 = np.genfromtxt('daten2.txt', unpack=True)

T1 = T1+273.15
I1 = -I1
diff = np.array([])
for i in range(len(T1)-1):  # geht länge des arrays durch
    d = T1[i]-T1[i+1]
    diff = np.append(diff, d)

print(diff)

print(np.mean(diff))
print(stats.sem(diff))

plt.plot(T1, I1, 'rx', label='Messwerte')
plt.xlabel('T in K')
plt.ylabel('I in pA')
plt.grid()
plt.legend(loc='best')
plt.savefig('kurve2.pdf')
