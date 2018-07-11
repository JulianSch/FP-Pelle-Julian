import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.stats as stats
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix, ufloat
import uncertainties.unumpy as unp


A = np.matrix([[0,np.sqrt(2),0,np.sqrt(2),0,0,0,0,0],[0,0,np.sqrt(2),0,np.sqrt(2),0,np.sqrt(2),0,0],[0,0,0,0,0,np.sqrt(2),0,np.sqrt(2),0],[1,1,1,0,0,0,0,0,0],[0,0,0,1,1,1,0,0,0],[0,0,0,0,0,0,1,1,1],[0,np.sqrt(2),0,0,0,np.sqrt(2),0,0,0],[np.sqrt(2),0,0,0,np.sqrt(2),0,0,0,np.sqrt(2)],[0,0,0,np.sqrt(2),0,0,0,np.sqrt(2),0],[0,0,1,0,0,1,0,0,1],[0,1,0,0,1,0,0,1,0],[1,0,0,1,0,0,1,0,0]])
print('A=',A)

ATA = np.matrix.dot(np.matrix.transpose(A),A)
print('ATA=', ATA)
print('ATA^-1=', np.linalg.inv(ATA))
print('varianzfortpflanzung: ', np.diagonal(np.linalg.inv(ATA)))
C = 0.03*np.eye(9,9)
print('C=', C)
sigma =  np.matrix.dot(ATA,C)
print('sigam=',sigma)

N=(np.diagonal(sigma))**(-2)
print('N=',N)
