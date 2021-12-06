from cell_io import load_cells


# loads a list of cell objects 
cells = load_cells()

cell = cells[0]
print(cell)
print("Available cell keys: ",list(cell.keys()))
print("Available cycle keys: ",list(cell[0].keys()))

# can access cell attrs like dic
print("min cell SOC: ",cell["SOC_min"])

# can use list comprehensions to get cycle elements
d_cap = [cycle["dis_c"] for cycle in cell]

print("First 10 cycles discharge cap: ",d_cap[:10])

print("First cycle time: ",cell[0]["cycle_t"])