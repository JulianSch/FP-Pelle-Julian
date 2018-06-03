import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import uncertainties.unumpy as unp

x, y = np.genfromtxt('kontrastplot.txt', unpack=True)
xrad=(((2*np.pi)/360)*x)
a=np.linspace(-0.36,0.75,1000)
a=((np.pi)*a)
alpha=0.85
beta =6
gamma =(((2*np.pi)/360)*(20))
delta=-0.3
def K(xrad,alpha, beta, gamma, delta):
     return (np.abs(alpha*np.sin(beta*xrad+gamma))+delta)
params, covariance = curve_fit(K, xrad, y)
errors = np.sqrt(np.diag(covariance))
print('U =', params[0], '±', errors[0])
print('q =', params[1], '±', errors[1])
print('phi =', params[2], '+/-', errors[2])
print('abs =', params[3], '+/-', errors[3])
plt.figure()
plt.plot(a,K(a,*params))
plt.plot(xrad, y, 'rx')
plt.grid(True)
plt.xlabel(r'$Winkel$ $in$ $Rad$')
plt.ylabel(r'$Kontrast$')
plt.title(r"$Kontrast$ $der$ $Apparatur$")
#plt.legend(loc='best')
plt.tight_layout()
plt.savefig('kontrastplott.pdf')
