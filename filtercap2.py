from cell_io import load_cells,load_times
import pickle

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline

PLOT = True

# loads a list of cell objects 
cells = load_cells()

cells = [cell for cell in cells if cell["chem"] == "NMC"][1:]
# cells = [cell for cell in cells if cell["SOC_min"] == 0]


i = 0
for cell in cells:
	inds = [cell.cycles.index(cycle) for cycle in cell if cycle["dis_c"] < 3.2 and cycle["dis_c"] > 2 and cycle["min_v"] < 2.2]

	cycles = [cell[ind] for ind in inds]
	d_cap = [cycle["dis_c"] for cycle in cell]
	a_d_cap = [elem["dis_c"] for elem in cycles]

	soc_range = "{SOC_min}-{SOC_max}".format(SOC_min=cell["SOC_min"],SOC_max=cell["SOC_max"])
	t_time = [cycle["tes_t"] for cycle in cycles]
	ah_thru = np.array([cycle["dis_c"]+ cycle["cha_c"] for cycle in cell])
	ah_vec = np.cumsum(ah_thru)
	a_ah_vec = [ah_vec[elem] for elem in inds]
	split = False
	if cell["SOC_min"] == 0 and not i == 6:
		split = True
		a_ah_vec = a_ah_vec[::4]
		a_d_cap = a_d_cap[::4]
		s = len(a_d_cap)/1500
		a_ah1 = a_ah_vec[:int(len(a_ah_vec)/4)]
		a_ah2 = a_ah_vec[len(a_ah1):]
		spl1 = UnivariateSpline(np.array(a_ah1),np.array(a_d_cap[:len(a_ah1)]),s=s,ext=3)
		spl2 = UnivariateSpline(np.array(a_ah2),np.array(a_d_cap[len(a_ah1):]),s=s,ext=3)
		# spline = UnivariateSpline(np.array(a_ah_vec),np.array(a_d_cap),s=s,ext=0)
		s_d_cap1 = spl1(a_ah1)
		s_d_cap2 = spl2(a_ah2)
		s_d_cap = np.append(s_d_cap1,s_d_cap2,0)
		spline = UnivariateSpline(np.array(a_ah_vec),s_d_cap,s=0.05)
		s_d_cap = spline(a_ah_vec)
	else:
		s = 0.2
		spline = UnivariateSpline(np.array(a_ah_vec),np.array(a_d_cap),s=s,ext=0)
		s_d_cap = spline(a_ah_vec)

	if i == 6:
		a_ah_vec = a_ah_vec[::10]
		a_d_cap = a_d_cap[::10]
		s = .06
		spline = UnivariateSpline(np.array(a_ah_vec),np.array(a_d_cap),s=s,ext=0)
		s_d_cap = spline(a_ah_vec)

	# ah = 0
	# ah_vec = []
	# for cycle in cell:
	# 	ah += cycle["dis_c"] + cycle["cha_c"]
	# 	ah_vec.append(ah)
	# 	cap = float(spline(ah))

	# 	cycle["f_cap"] = cap
	
	# caps = [cycle["dis_c"] for cycle in cell]
	# fcaps = [cycle["f_cap"] for cycle in cell]
	# # plt.plot(ah_vec,caps,linestyle='dashed')
	# plt.plot(fcaps,linestyle='dashed')
	# plt.show()
	
	


	if PLOT:
		plt.figure()
		plt.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
		chem = cell["chem"]
		# plt.plot(ah_vec,d_cap,label="raw")
		plt.plot(a_ah_vec,a_d_cap,label="filtered")
		plt.plot(a_ah_vec,s_d_cap,label="spline on filtered")
		plt.title(f"SOC range: {soc_range} i: {i}")
		plt.xlabel("Amp-hour Throughput [Ah]")
		plt.ylabel("Capacity [Ah]")
		plt.ylim((2,3.5))
		# print(len(d_cap),len(ah_vec))
		plt.legend(loc=1)
		i += 1
		# # plt.savefig(f"NMC{i}.png",
		# 		format = "png",
		# 		dpi = 128,)
		plt.show()


# path = "cells.p"
# pickle.dump(cells,open(path,'wb+'))

# d_caps,t_times, full_d,full_t, del_ts, inds = load_times()
# d_caps = np.array(d_caps)
# t_times = np.array(t_times)
# plt.figure()
# # plt.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)

# plt.plot(np.array(del_ts)/max(del_ts),label="Sample Period")
# plt.plot(inds,np.array(d_caps)/max(d_caps),label="Capacity")
# plt.title("Normalized Capacity vs Sample Period")
# plt.ylabel("Normalized Magnitudes")
# plt.xlabel("Step Number")
# plt.legend(loc="upper right")

# plt.show()
