import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties as unc
import uncertainties.unumpy as unp
from numpy import linspace, pi,exp, sin, cos, tan, loadtxt, savetxt, mean, zeros, size, log, sum, sqrt, arctan
from uncertainties import ufloat
import scipy.stats as st
from scipy.constants import c, e, h
from astropy.io import ascii
from scipy.integrate import quad
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 13
k = 1.3806504*10**(-23)
T = 263.75
W = 109 *10**(-21)
d_W = 2 *10**(-21)
b = 4807*10**(-5)
d_b = 10 * 10**(-5)
tau = (T**2 * k)/(W*b) * np.exp((-W)/(k*T))
tau_fehler = np.sqrt(
((-k*T)/(W*b**2)*np.exp((-W)/(k*T))*d_b)**2
+
((-T*(W+k*T))/(b*W**2)*np.exp((-W)/(k*T))*d_W)**2
)
print("tau",tau)
print("tau_fehler",tau_fehler)
