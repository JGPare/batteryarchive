# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 12:14:08 2021

@author: Jules Pare
"""

import numpy as np
from datetime import datetime
import math

from cell_io import load_pcells
from fit_check import cfit


def print_green(message):
	print("\033[92m" + str(message) + "\033[0m")

def print_yellow(message):
	print("\033[93m" + str(message) + "\033[0m")

def print_red(message):
	print("\033[91m" + str(message) + "\033[0m")



cells = load_pcells()[:10]

def func(x, c1,c2,c3):
	Ah = np.array(x[0,:])
	Crate = np.array(x[1,:])
	return c1*np.exp(c2*Crate)*Ah**c3

x = []
y = []

i = 0
for cell in cells:

	ah_thru = np.array([cycle["dis_c"]+ cycle["cha_c"] for cycle in cell])
	ah_vec = np.cumsum(ah_thru)
	dcap = np.array([cycle["f_cap"] for cycle in cell])
	ah_vec[0] = 0

	d_c_rate = cell["d_c_rate"]

	x_t = np.ones([2,len(ah_vec)])
	x_t[0,:] = ah_vec
	x_t[1,:] *= d_c_rate

	y_t = dcap[0] - dcap
	x.append(x_t)
	y.append(y_t)


try:
	print("\n\n   Testing ...\n")
	start_time = datetime.now()

	cf_kwargs = {"bounds": (-5,5)}

	# Check that cross and single mode both produce same results
	# for the case where cell index 0 is used to access fit
	fit_list,avg = cfit(func,x,y,cf_kwargs,mode="cross",prt=False)
	fit1 = fit_list[0]
	fit1_ = cfit(func,x,y,cf_kwargs,mode=0,prt=False)
	fit_all = cfit(func,x,y,cf_kwargs,mode="all",prt=False)

	assert fit1.keys() == fit1_.keys()
	assert fit_all.keys() == fit1.keys()
	assert fit1["y_a"][15] == fit1_["y_a"][15]
	assert math.isclose(fit1["Radj"],fit1_["Radj"])
	assert math.isclose(fit1["popt"][0],fit1_["popt"][0])


	try:
		print("check mismatched vector length")
		cfit(func,x[1:],y,cf_kwargs,mode=0,prt=False)
	except ValueError:
		print("mismatched vector length caught")
		pass


except:
	test_time = (datetime.now() - start_time).total_seconds()
	print(40 * " ", end="\r", flush=True)
	print("")
	print_red("   A test has failed! ({} s). Details:\n".format(round(test_time, 6)))
	raise
else:
	test_time = (datetime.now() - start_time).total_seconds()
	print(40 * " ", end="\r", flush=True)
	print("")
	print_green("   All tests have passed. ({} s)\n".format(round(test_time, 6)))