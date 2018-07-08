import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.stats as stats
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix, ufloat
import uncertainties.unumpy as unp


A = np.matrix([[4,1,1,1,2,0,1,0,2],[1,6,1,2,1,2,0,1,0],[1,1,4,0,2,1,2,0,1],[1,2,0,6,1,1,1,2,0],[2,1,2,1,6,1,2,1,2],[0,2,1,1,1,6,0,2,1],[1,0,2,1,2,0,4,1,1],[0,1,0,2,1,2,1,6,1],[2,0,1,0,2,1,1,1,4]])

print('ATA=',A)

A= np.linalg.inv(A)
print('A=',A)
