import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statistics import mean

from cell_io import load_pcells
from fit_check import cfit, c_plot, r_plot

cells = load_pcells()[:10]

def func(x, c1,c2,c3):
		R = 8.3145
		Ah = np.array(x[0,:])
		Crate = np.array(x[1,:])
		T = np.array(x[2,:])
		dod = np.array(x[3,:])
		return c1*np.exp(c2*Crate)*Ah**c3 

x = []
y = []

i = 0
for cell in cells:

	ah_thru = np.array([cycle["dis_c"]+ cycle["cha_c"] for cycle in cell])
	ah_vec = np.cumsum(ah_thru)
	dcap = np.array([cycle["f_cap"] for cycle in cell])
	ah_vec[0] = 0

	T_a = cell["T_a"]
	d_c_rate = cell["d_c_rate"]
	dod = int(cell["SOC_max"]) - int(cell["SOC_min"])

	x_t = np.ones([4,len(ah_vec)])
	x_t[0,:] = ah_vec
	x_t[1,:] *= d_c_rate
	x_t[2,:] *= T_a
	x_t[3,:] *= dod

	y_t = dcap[0] - dcap
	x.append(x_t)
	y.append(y_t)


# flist,avgs = cfit(func,x,y,{"bounds":(-5,5)})
fit = cfit(func,x,y,{"bounds":(-5,5)},mode="all")
fit_list,avg = cfit(func,x,y,{"bounds":(-5,5)},mode="cross")

print(fit["popt"],avg["popt"])




