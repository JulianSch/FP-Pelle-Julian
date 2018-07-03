import numpy as np
import uncertainties.numpy as unp

n, x = np.genfromtxt('daten3.txt', unpack=True)
data = np.array([n, x])

np.savetxt('tabelle.txt', np.column_stack([n, x]))
np.savetxt('TexTabellen/luft.txt', np.column_stack([
        unp.nominal_values(N),
#        unp.std_devs(N),
        unp.nominal_values(n_array),
#        unp.std_devs(n_array),
        ]), delimiter=' & ', newline=r' \\'+'\n',
fmt='%.0f & %.0f & %.7f & %.7f')
# with open('test.txt', 'r') as f:
    # print(f.read())
#nicht fertig steht nur bullshit drin
