import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from cell_io import load_cells,load_times,load_pcells
from fit_check import cfit

cells = load_pcells()

cell = cells[1]

ah_thru = np.array([cycle["dis_c"]+ cycle["cha_c"] for cycle in cell])
ah_vec = np.cumsum(ah_thru)
dcap = np.array([cycle["dis_c"] for cycle in cell])

T_a = cell["T_a"]
d_c_rate = cell["d_c_rate"]

x = np.ones([3,len(ah_vec)])
x[0,:] = ah_vec
x[1,:] *= d_c_rate
x[2,:] *= T_a

y = dcap[0] - dcap

def func(x, c1,c2, c4):
	R = 8.3145
	Ah = np.array(x[0,:])
	Crate = np.array(x[1,:])
	T = np.array(x[2,:])
	return c1*(Ah)**c4 + c2

popt, pcov = curve_fit(func, x, y,bounds=(0,5))


# Where xplt is the xaxis plot data, and actual is the yaxis measured data

actual = y
xplt = ah_vec

cfit(func,x,popt,pcov,xplt,actual)



