import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.stats as stats
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix


for i in range(0, 17, 1):
    t = (temp[i] - temp[i+1])
    s = (strom[i] + strom[i+1])/2
    a += np.sqrt((s*t)**2)
print('Modus 0')
return a
