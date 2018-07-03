import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
import scipy.constants as const
from scipy.optimize import curve_fit
from uncertainties import correlated_values, correlation_matrix
from uncertainties import ufloat
from uncertainties.unumpy import (nominal_values as noms, std_devs as stds)

np.savetxt('TexTabellen/luft.txt', np.column_stack([
        unp.nominal_values(N),
        unp.std_devs(N),
        unp.nominal_values(n_array),
        unp.std_devs(n_array),
        ]), delimiter=' & ', newline=r' \\'+'\n',fmt='%.0f & %.0f & %.7f & %.7f')
