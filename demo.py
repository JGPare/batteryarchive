import numpy as np
import matplotlib.pyplot as plt

from cell_io import load_pcells
from fit_check import cfit

cells = load_pcells()[:10]

# define function
def func(x, c1,c2,c3):
		Ah = np.array(x[0,:])
		Crate = np.array(x[1,:])
		return c1*np.exp(c2*Crate)*Ah**c3

x = []
y = []


# Generate x,y dataset
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


# fit function to data
cf_kwargs = {"bounds":(-5,5)}
fit = cfit(func,x,y,cf_kwargs,mode="all")
fit_list,avg = cfit(func,x,y,cf_kwargs,mode="cross")

print("Fit coeffs using all data:",fit["popt"])
print("Fit coeffs averaged from all cross analysis fits:",avg["popt"])

# plotting stuff for fit on cell 0
y0 = cells[0][0]["f_cap"]

f1 = fit_list[0]
# get ah from x input
xplt = f1["x"][0]
cap_a = y0 - f1["y_a"]
cap_p = y0 - f1["y_p"]
cap_l = y0 - f1["y_min"]
cap_u = y0 - f1["y_max"]
R_sq = f1["R2"]
R_adj = f1["Radj"]

fig, (ax1,ax2) = plt.subplots(1,2,figsize=(16,6), dpi= 100, facecolor='w', edgecolor='k')
ax1.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
ax1.plot(xplt,cap_a,zorder=2,label="Actual")
ax1.plot(xplt,cap_p,zorder=2,label="Predicted")
ax1.fill_between(xplt,cap_l, cap_u,color = 'black', alpha = 0.15,label="Confidence interval")
ax1.set_title("Cell 0 Fit Check")
ax1.set_xlabel(r"Ah Throughput [Ah]")
ax1.set_ylabel(r"Capacity [Ah]")
_,x_end = plt.xlim()
y_str,y_end = plt.ylim()
y_s = (y_end-y_str)/2+y_str
s = (f"R2: {100*R_sq:.2f} %\n"
    f"R2_adj: {100*R_adj:.2f} %")
# ax1.text(x_end*.75,y_s,s,fontsize=10,color='r')
ax1.legend()

res = f1["res"]
s_res = np.std(res)
std_res = res/s_res
# Standard residuals plot, used to check for outliers in the data, etc
ax2.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
ax2.plot(std_res,label="Residual",linestyle="",marker="x")
ax2.plot([0,len(std_res)],[0,0],linestyle="dashed",color="k")
ax2.set_title("Standard Residuals Plot")
ax2.set_xlabel("Data Point")
ax2.set_ylabel("Standard Deviations")

plt.show()




