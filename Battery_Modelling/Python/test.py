#
#   Test script for battery degradation models
#


import battery_degr_models as bdm
import math
import matplotlib.pyplot as plt
import numpy as np
import random


# ---- tests ----


try:
    # 1. test functionality of get_half_cycles() on sinusoidal input
    rand_amp = random.uniform(1, 4)
    rand_const = random.uniform(-2 * rand_amp, 2 * rand_amp)
    T = 12
    input_array = np.zeros(20 * T + 1)
    for i in range(0, len(input_array)):
        input_array[i] = rand_amp * math.sin((2 * math.pi * i) / T) + rand_const

    idx_array = bdm.get_half_cycles(input_array)

    for i in range(0, len(idx_array)):
        bool = (
            idx_array[i] % T == 0 or
            idx_array[i] % T == T / 4 or
            idx_array[i] % T == (3 * T) / 4
        )
        assert bool


    # 2. test functionality of get_cycle_depths() on sinusoidal input
    depths_array = bdm.get_cycle_depths(input_array, idx_array)

    assert (len(depths_array) == len(idx_array) - 1)
    for i in range(0, len(depths_array)):
        bool = (
            abs(depths_array[i] - rand_amp) < 1e-9 or
            abs(depths_array[i] - 2 * rand_amp) < 1e-9
        )
        assert bool


    # 3. test functionality of get_half_cycles() on complicated input
    input_array_2 = np.genfromtxt(
        "../data/test/Yuquot_load_model_kW_30yr_dt-1hr.csv",
        delimiter=",",
        skip_header=1
    )
    rand_idx = random.randrange(0, np.size(input_array_2, 0) - 1 - 1000)
    input_array_2 = input_array_2[rand_idx : rand_idx + 1000, 1]
    input_array_2 = (
        (1 / (np.max(input_array_2) - np.min(input_array_2))) *
        (input_array_2 - np.min(input_array_2))
    )

    idx_array_2 = bdm.get_half_cycles(input_array_2)


    # 4. test functionality of get_cycle_depths() on complicated input
    depths_array_2 = bdm.get_cycle_depths(input_array_2, idx_array_2)

    assert (len(depths_array_2) == len(idx_array_2) - 1)
    assert (np.max(depths_array_2) <= 1)
    assert (np.min(depths_array_2) >= 0)


    # 5. test functionality of deshpande_bernardi() on complicated input
    params = np.ones(3)
    t_hrs = random.uniform(1000, 2000)
    time_array_hrs = np.linspace(0, t_hrs, int(t_hrs + 1))
    SOC_array = (
        (1 / (np.max(input_array_2) - np.min(input_array_2))) *
        (input_array_2 - np.min(input_array_2))
    )

    Q_loss_kWh = bdm.deshpande_bernardi(params, time_array_hrs, SOC_array)

    assert (not np.isnan(Q_loss_kWh))
    assert (Q_loss_kWh >= 0)


    # 6. test functionality of get_half_cycles() on ramp and plateau
    ramp_and_plateau = np.asarray([])
    for i in range(0, 20):
        lower_plateau = np.zeros(random.randrange(10, 50))
        ramp_and_plateau = np.append(ramp_and_plateau, lower_plateau)

        plat = random.uniform(0, 1)
        up_ramp = np.linspace(0, plat, random.randrange(10, 50))
        ramp_and_plateau = np.append(ramp_and_plateau, up_ramp)

        upper_plateau = plat * np.ones(random.randrange(10, 50))
        ramp_and_plateau = np.append(ramp_and_plateau, upper_plateau)

        down_ramp = np.linspace(plat, 0, random.randrange(10, 50))
        ramp_and_plateau = np.append(ramp_and_plateau, down_ramp)

    idx_array_3 = bdm.get_half_cycles(ramp_and_plateau)


    # 7. test functionality of get_cycle_depths() on ramp and plateau
    depths_array_3 = bdm.get_cycle_depths(ramp_and_plateau, idx_array_3)

    assert (len(depths_array_3) == len(idx_array_3) - 1)
    assert (np.max(depths_array_3) <= 1)
    assert (np.min(depths_array_3) >= 0)


    # 8. test functionality of quad_approx()
    params = np.ones(10)
    cap_0_kWh = 100
    charge_current_array_A = SOC_array
    discharge_current_array_A = SOC_array
    ambient_temp_avg_K = 293

    cap_1_kWh = bdm.quad_approx(
        params,
        cap_0_kWh,
        time_array_hrs,
        SOC_array,
        charge_current_array_A,
        discharge_current_array_A,
        ambient_temp_avg_K
    )

    assert (not np.isnan(cap_1_kWh))
    assert (cap_1_kWh <= cap_0_kWh)
    assert (cap_1_kWh >= 0)


except:
    print("\t***A test has failed***\n")
    raise


else:
    print("\t***All tests passed***")


# ---- test plots ----


# for tests (1) and (2)
#"""
plt.figure(figsize=(8, 6))
plt.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
plt.plot(input_array, zorder=2)
for i in range(0, len(idx_array) - 1):
    y_plot = input_array[idx_array[i] : idx_array[i + 1] + 1]
    x_plot = np.linspace(idx_array[i], idx_array[i + 1], len(y_plot))
    plt.plot(x_plot, y_plot, zorder=3)
plt.xlabel(r"$t$")
plt.ylabel(r"$f(t)$")
plt.title(r"Test Plot for Tests (1) and (2)")
#"""

# for tests (3) and (4)
#"""
plt.figure(figsize=(8, 6))
plt.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
plt.plot(input_array_2, zorder=2)
for i in range(0, len(idx_array_2) - 1):
    y_plot = input_array_2[idx_array_2[i] : idx_array_2[i + 1] + 1]
    x_plot = np.linspace(idx_array_2[i], idx_array_2[i + 1], len(y_plot))
    plt.plot(x_plot, y_plot, zorder=3)
plt.xlabel(r"$t$")
plt.ylabel(r"$f(t)$")
plt.title(r"Test Plot for Tests (3) and (4)")
#"""

# for tests (6) and (7)
#"""
plt.figure(figsize=(8, 6))
plt.grid(which="both", axis="both", color="C7", alpha=0.5, zorder=1)
plt.plot(ramp_and_plateau, zorder=2)
for i in range(0, len(idx_array_3) - 1):
    y_plot = ramp_and_plateau[idx_array_3[i] : idx_array_3[i + 1] + 1]
    x_plot = np.linspace(idx_array_3[i], idx_array_3[i + 1], len(y_plot))
    plt.plot(x_plot, y_plot)
plt.xlabel(r"$t$")
plt.ylabel(r"$f(t)$")
plt.title(r"Test Plot for Tests (6) and (7)")
#"""

plt.show()
