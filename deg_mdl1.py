import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statistics import mean

from cell_io import load_cells,load_times,load_pcells
from fit_check import cfit, c_plot, r_plot

cells = load_pcells()

cstart = 0
cend = 10

R2_vec = []
Radj_vec = []

def func(x, c1,c2,c3,c4,c5):
		R = 8.3145
		Ah = np.array(x[0,:])
		Crate = np.array(x[1,:])
		T = np.array(x[2,:])
		dod = np.array(x[3,:])
		# return c1*(c3*Crate)*Ah**c4 
		return c1*np.exp(c2*Crate)*Ah**c3 + c4*dod**c5*Ah

for j in range(cstart,cend):

	mdl_cells = cells[:cend+1]
	prd_cells = [mdl_cells.pop(j)]

	i = 0
	for cell in mdl_cells:

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

		if i == 0:
			x = x_t
			y = np.array(y_t)
		else:
			x = np.concatenate((x_t,x),1)
			y = np.concatenate((y_t,y))

		i += 1

	popt, pcov = curve_fit(func, x, y,bounds=(-5,5))

	cell = prd_cells[0]
	ah_thru = np.array([cycle["dis_c"]+ cycle["cha_c"] for cycle in cell])
	ah_vec = np.cumsum(ah_thru)
	ah_vec[0] = 0
	dcap = np.array([cycle["f_cap"] for cycle in cell])
	T_a = cell["T_a"]
	d_c_rate = cell["d_c_rate"]
	dod = int(cell["SOC_max"]) - int(cell["SOC_min"])

	x = np.ones([4,len(ah_vec)])
	x[0,:] = ah_vec
	x[1,:] *= d_c_rate
	x[2,:] *= T_a
	x[3,:] *= dod

	y = dcap[0] - dcap

	# Where xplt is the xaxis plot data, and actual is the yaxis measured data

	actual = y
	xplt = ah_vec

	# popt = [0.003,0.04506,0.56694,0.001,0.0469]

	actual,predict,lower,upper,res,R_sq,R_adj = cfit(func,x,popt,pcov,actual,False)

	R2_vec.append(abs(R_sq))
	Radj_vec.append(abs(R_adj))

	cap_a = dcap[0] - actual
	cap_p = dcap[0] - predict
	cap_l = dcap[0] - lower
	cap_u = dcap[0] - upper

	# plt.figure()
	# plt.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
	# plt.plot(xplt,cap_a,zorder=2,label="Actual")
	# plt.plot(xplt,cap_p,zorder=2,label="Predicted")
	# plt.fill_between(xplt,cap_l, cap_u,color = 'black', alpha = 0.15,label="Confidence interval")
	# plt.title(f"Cell {j} Fit Check")
	# plt.xlabel(r"Ah Throughput [Ah]")
	# plt.ylabel(r"Capacity [Ah]")
	# _,x_end = plt.xlim()
	# y_str,y_end = plt.ylim()
	# y_s = (y_end-y_str)/2+y_str
	# s = (f"R2: {100*R_sq:.2f} %\n"
	# 	f"R2_adj: {100*R_adj:.2f} %")
	# plt.text(x_end*.75,y_s,s,fontsize=10,color='r')
	# plt.legend()
	# plt.show()

print(mean(R2_vec))
print(mean(Radj_vec))




