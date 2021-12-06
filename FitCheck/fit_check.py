
import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

def cfit(func,x,popt,pcov,xplt,actual):

	p_sigma = np.sqrt(np.diag(pcov))

	predict = func(x,*popt)
	upper = func(x,*(popt + p_sigma))
	lower = func(x,*(popt - p_sigma))

	# method checked against results from https://ncalculators.com/statistics/r-squared-calculator.htm
	res = actual-predict
	s_res = np.std(res)
	std_res = res/s_res
	corr_matrix = np.corrcoef(actual, predict)
	corr = corr_matrix[0,1]
	R_sq = corr**2

	n = len(actual)
	k = len(np.diag(pcov))

	# adjusted to account for number of variables, used to prevent overfitting
	R_adj = 1 - ((1-R_sq**2)*(n-1))/(n-k-1)

	print("\nFIT SUMMARY:")
	print(f"\nR-squared Error: {100*R_sq:.3f} %\n"
		f"R-adjusted Error: {100*R_adj:.3f} %\n\n"
		"Parameter Values:\n")
	for i in range(len(popt)):
		p = popt[i]
		p_err = p_sigma[i]*100
		print(f"c{i} = {p:.3f} +- {p_err:.5f} %")
	print()

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


	# Standard residuals plot, used to check for outliers in the data, etc
	plt.figure()
	plt.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
	plt.plot(std_res,label="Residual",linestyle="",marker="x")
	plt.plot([0,len(actual)],[0,0],linestyle="dashed",color="k")
	plt.title("Residuals Plot")
	plt.xlabel("Data Point")
	plt.ylabel("Error [ ]")
	plt.show()

