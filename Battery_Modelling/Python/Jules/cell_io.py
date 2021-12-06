import os
import numpy as np
import csv
from datetime import datetime
filelist = [f for f in os.listdir(os.getcwd()) if f.endswith("cycle_data.csv")]


class cell():
	def __init__(self,attr,cycles):
		self.attr = attr
		self.cycles = cycles

	def __iter__(self):
		return iter(self.cycles)

	def __getitem__(self,item):
		if type(item) == int:
			return self.cycles[item]
		if type(item) == str:
			return self.attr[item]
	def __len__(self):
		return len(self.cycles)

	def __str__(self):
		line = "_"*79
		form = self["form"]
		chem = self["chem"]
		T_a = self["T_a"]
		c_c_rate = self["c_c_rate"]
		d_c_rate = self["d_c_rate"]
		SOC_min = self["SOC_min"]
		SOC_max = self["SOC_max"]
		return (f"\n{line}\n\n"
				f"Cell Form: {form}\n"
				f"Cell Chemisty: {chem}\n"
				f"Ambient Temp: {T_a:.3f} [deg/C]\n"
				f"Number of Cycles: {len(self)}\n"
				f"Charging C-rate: {c_c_rate:.3f} []\n"
				f"Discharging C-rate: {d_c_rate:.3f} []\n"
				f"SOC min: {SOC_min:.1f} %\n"
				f"SOC max: {SOC_max:.1f} %\n")

	def keys(self):
		return self.attr.keys()
				

def load_cells():
	cells = []

	for file in filelist:

		attr = {}

		s_file = file.split('_')
		attr["org"] = s_file[0]
		attr["form"] = s_file[1]
		attr["chem"] = s_file[2]
		attr["T_a"] = int(''.join([elem for elem in s_file[3] if elem.isnumeric()]))
		soc_range = s_file[4].split('-')
		attr["SOC_min"] = int(soc_range[0])
		attr["SOC_max"] = int(soc_range[1])
		c_rate = s_file[5].split('-')
		attr["c_c_rate"] = float(''.join([elem for elem in c_rate[0] if not elem.isalpha()]))
		attr["d_c_rate"] = float(''.join([elem for elem in c_rate[1] if not elem.isalpha()]))

		with open(file, newline='') as f:
			reader = csv.reader(f)
			header = next(reader)
			key_list = []
			for col in header:
				key = col.lower().split('_')
				key = key[0][0:3] + '_' + key[1][0]
				key_list.append(key)
			
			cycles = []
			for row in reader:
				data = {}
				i = 0
				for key in key_list:
					if i == 0:
						data[key] = int(float(row[i]))
					elif i in [1,2]:
						data[key] =  datetime.strptime(row[i],"%Y-%m-%d %H:%M:%S.%f")
					else:
						data[key] = float(row[i])
					i += 1
				data["cycle_t"] = data["end_t"]-data["sta_t"]
				cycles.append(data)
		
		c = cell(attr,cycles)
		cells.append(c)

	return cells






