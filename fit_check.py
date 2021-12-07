
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean

from scipy.optimize import curve_fit



def app_fit(func,x,popts):
	y_vec = []
	for popt in popts:
		y_vec.append(func(x,*popt))
	return y_vec


# for use with multiple x->y datasets
def cfit(func,x,y,cf_kwargs,mode="cross",prt=True):

	if len(x) != len(y):
		print(f"X vector length {len(x)} does not match Y vector length {len(y)}")
		raise ValueError
	if mode == "cross":

		fit_list = []
		R2_vec = []
		Radj_vec = []
		m_err_vec = []
		mr_err_vec = []

		for j in range(len(x)):

			d_fit = {}
			x_test = x[j]
			x_fit = np.hstack([*x[:j],*x[j+1:]])
			y_test = y[j]
			y_fit = np.hstack([*y[:j],*y[j+1:]])

			popt, pcov = curve_fit(func,x_fit,y_fit,**cf_kwargs)
			p_sigma = np.sqrt(np.diag(pcov))

			d_fit["x"] = x_test
			d_fit["y_a"] = y_test
			d_fit["popt"] = popt
			d_fit["pcov"] = pcov
			d_fit["p_sigma"] = p_sigma

			popts = [popt,popt + p_sigma,popt - p_sigma]
			predict,upper,lower = app_fit(func,x_test,popts)
			actual = y_test

			d_fit["y_p"] = predict
			d_fit["y_max"] = upper
			d_fit["y_min"] = lower

			# method checked against results from https://ncalculators.com/statistics/r-squared-calculator.htm
			res = actual-predict

			d_fit["res"] = res
			m_err = max(abs(res))
			mr_err = max([abs(res[k]/actual[k]) for k in range(len(actual)) if actual[k] != 0])

			d_fit["m_err"] = m_err
			d_fit["mr_err"] = mr_err

			s_res = np.std(res)
			std_res = res/s_res
			d_fit["std_res"] = std_res

			y_bar = np.mean(actual)
			ssr = np.sum(res**2)
			sst = np.sum((actual-y_bar)**2)
			R2 = 1-(ssr/sst)

			d_fit["R2"] = R2
			n = len(actual)
			k = len(np.diag(pcov))

			# adjusted to account for number of variables, used to prevent overfitting
			Radj = 1 - ((1-R2**2)*(n-1))/(n-k-1)
			d_fit["Radj"] = Radj

			f_str = (
				"\nCELL {j} FIT SUMMARY:"
				f"\nR-squared Fit: {100*R2:.3f} %\n"
				f"R-adjusted Fit: {100*Radj:.3f} %\n\n"
				f"Max Error: {m_err:.3f} \n"
				f"Max Relative Error: {mr_err*100:.3f} %\n\n"
				"Parameter Values:\n"
				)

			for i in range(len(popt)):
				p = popt[i]
				p_err = p_sigma[i]*100
				f_str += f"c{i+1} = {p:.3f} +- {p_err:.5f} %\n"

			d_fit["str"] = f_str
			fit_list.append(d_fit)
			R2_vec.append(R2)
			Radj_vec.append(Radj)
			m_err_vec.append(m_err)
			mr_err_vec.append(mr_err)

		avg_R2 = mean(R2_vec)
		avg_Radj = mean(Radj_vec)
		avg_m_err = mean(m_err_vec)
		avg_mr_err = mean(mr_err_vec)
		avg_popt = np.mean([test["popt"] for test in fit_list],0)
		avgs = {"R2" : avg_R2,
			"Radj" : avg_Radj,
			"m_err" : avg_m_err,
			"mr_err" : avg_mr_err,
			"popt" : avg_popt}

		if prt:
			print("\nFIT SUMMARY"
				f"\nAvg R2 Fit: {100*avg_R2:.3f} %"
				f"\nAvg Radj Fit: {100*avg_Radj:.3f} %"
				f"\nAvg Max Error: {avg_m_err:.3f} "
				f"\nAvg Max Relative Error: {100*avg_mr_err:.3f} %\n"
				)
		return fit_list,avgs

	elif type(mode) == int:
		if mode > len(x):
			print("Fit index exceeds input list length")
			return
		j = mode
		d_fit = {}
		x_test = x[j]
		x_fit = np.hstack([*x[:j],*x[j+1:]])
		y_test = y[j]
		y_fit = np.hstack([*y[:j],*y[j+1:]])

		popt, pcov = curve_fit(func,x_fit,y_fit,**cf_kwargs)
		p_sigma = np.sqrt(np.diag(pcov))

		d_fit["x"] = x_test
		d_fit["y_a"] = y_test
		d_fit["popt"] = popt
		d_fit["pcov"] = pcov
		d_fit["p_sigma"] = p_sigma

		popts = [popt,popt + p_sigma,popt - p_sigma]
		predict,upper,lower = app_fit(func,x_test,popts)
		actual = y_test

		d_fit["y_p"] = predict
		d_fit["y_max"] = upper
		d_fit["y_min"] = lower

		# method checked against results from https://ncalculators.com/statistics/r-squared-calculator.htm
		res = actual-predict

		d_fit["res"] = res
		m_err = max(abs(res))
		mr_err = max([abs(res[k]/actual[k]) for k in range(len(actual)) if actual[k] != 0])

		d_fit["m_err"] = m_err
		d_fit["mr_err"] = mr_err

		s_res = np.std(res)
		std_res = res/s_res
		d_fit["std_res"] = std_res

		y_bar = np.mean(actual)
		ssr = np.sum(res**2)
		sst = np.sum((actual-y_bar)**2)
		R2 = 1-(ssr/sst)

		d_fit["R2"] = R2
		n = len(actual)
		k = len(np.diag(pcov))

		# adjusted to account for number of variables, used to prevent overfitting
		Radj = 1 - ((1-R2**2)*(n-1))/(n-k-1)
		d_fit["Radj"] = Radj

		f_str = (
			f"\nCELL {j} FIT SUMMARY:"
			f"\nR-squared Fit: {100*R2:.3f} %\n"
			f"R-adjusted Fit: {100*Radj:.3f} %\n\n"
			f"Max Error: {m_err:.3f} \n"
			f"Max Relative Error: {mr_err*100:.3f} %\n\n"
			"Parameter Values:\n"
			)
		for i in range(len(popt)):
			p = popt[i]
			p_err = p_sigma[i]*100
			f_str += f"c{i+1} = {p:.3f} +- {p_err:.5f} %\n"

		d_fit["str"] = f_str
		if prt:
			print(f_str)
		return d_fit

	elif mode == "all":
		d_fit = {}

		x_fit = np.hstack(x)
		y_fit = np.hstack(y)

		popt, pcov = curve_fit(func,x_fit,y_fit,**cf_kwargs)
		p_sigma = np.sqrt(np.diag(pcov))

		d_fit["x"] = x_fit
		d_fit["y_a"] = y_fit
		d_fit["popt"] = popt
		d_fit["pcov"] = pcov
		d_fit["p_sigma"] = p_sigma

		popts = [popt,popt + p_sigma,popt - p_sigma]
		predict,upper,lower = app_fit(func,x_fit,popts)
		actual = y_fit

		d_fit["y_p"] = predict
		d_fit["y_max"] = upper
		d_fit["y_min"] = lower

		# method checked against results from https://ncalculators.com/statistics/r-squared-calculator.htm
		res = actual-predict

		d_fit["res"] = res
		m_err = max(abs(res))
		mr_err = max([abs(res[k]/actual[k]) for k in range(len(actual)) if actual[k] != 0])

		d_fit["m_err"] = m_err
		d_fit["mr_err"] = mr_err

		s_res = np.std(res)
		std_res = res/s_res
		d_fit["std_res"] = std_res

		y_bar = np.mean(actual)
		ssr = np.sum(res**2)
		sst = np.sum((actual-y_bar)**2)
		R2 = 1-(ssr/sst)

		d_fit["R2"] = R2
		n = len(actual)
		k = len(np.diag(pcov))

		# adjusted to account for number of variables, used to prevent overfitting
		Radj = 1 - ((1-R2**2)*(n-1))/(n-k-1)
		d_fit["Radj"] = Radj

		f_str = (
			"\nCELL {j} FIT SUMMARY:"
			f"\nR-squared Fit: {100*R2:.3f} %\n"
			f"R-adjusted Fit: {100*Radj:.3f} %\n\n"
			f"Max Error: {m_err:.3f} \n"
			f"Max Relative Error: {mr_err*100:.3f} %\n\n"
			"Parameter Values:\n"
			)
		for i in range(len(popt)):
			p = popt[i]
			p_err = p_sigma[i]*100
			f_str += f"c{i+1} = {p:.3f} +- {p_err:.5f} %\n"

		d_fit["str"] = f_str
		if prt:
			print(f_str)
		return d_fit

	# 	if PLOT:
	# 		fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6), dpi= 100, facecolor='w', edgecolor='k')
	# 		ax1.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
	# 		ax1.plot(xplt,cap_a,zorder=2,label="Actual")
	# 		ax1.plot(xplt,cap_p,zorder=2,label="Predicted")
	# 		ax1.fill_between(xplt,cap_l, cap_u,color = 'black', alpha = 0.15,label="Confidence interval")
	# 		ax1.set_title(f"Cell {j} Fit Check")
	# 		ax1.set_xlabel(r"Ah Throughput [Ah]")
	# 		ax1.set_ylabel(r"Capacity [Ah]")
	# 		_,x_end = plt.xlim()
	# 		y_str,y_end = plt.ylim()
	# 		y_s = (y_end-y_str)/2+y_str
	# 		s = (f"R2: {100*R_sq:.2f} %\n"
	# 		    f"R2_adj: {100*R_adj:.2f} %")
	# 		# ax1.text(x_end*.75,y_s,s,fontsize=10,color='r')
	# 		ax1.legend()

	# 		s_res = np.std(res)
	# 		std_res = res/s_res
	# 		# Standard residuals plot, used to check for outliers in the data, etc
	# 		ax2.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
	# 		ax2.plot(std_res,label="Residual",linestyle="",marker="x")
	# 		ax2.plot([0,len(std_res)],[0,0],linestyle="dashed",color="k")
	# 		ax2.set_title("Standard Residuals Plot")
	# 		ax2.set_xlabel("Data Point")
	# 		ax2.set_ylabel("Standard Deviations")

	# 		plt.show()

	# print(mean(R2_vec),mean(Radj_vec))



def c_plot(xplt,actual,predict,lower,upper):
	plt.figure()
	plt.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
	plt.plot(xplt,actual,zorder=2,label="Actual")
	plt.plot(xplt,predict,zorder=2,label="Predicted")
	plt.fill_between(xplt,lower, upper,color = 'black', alpha = 0.15,label="Confidence interval")
	plt.title(r"Actual, Predicted, Confidence")
	plt.xlabel(r"X Data")
	plt.ylabel(r"Y Data")
	plt.legend()
	plt.show()


def r_plot(res):
	s_res = np.std(res)
	std_res = res/s_res
	# Standard residuals plot, used to check for outliers in the data, etc
	plt.figure()
	plt.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
	plt.plot(std_res,label="Residual",linestyle="",marker="x")
	plt.plot([0,len(std_res)],[0,0],linestyle="dashed",color="k")
	plt.title("Standard Residuals Plot")
	plt.xlabel("Data Point")
	plt.ylabel("Standard Deviations")
	plt.show()

