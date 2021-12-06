#
#   Module of battery degradation models and helper functions
#


import math
import numpy as np


# ---- helper functions ----


def get_half_cycles(input_array):
    # cast to numpy array
    input_array = np.asarray(input_array)

    # init idices array (to ID where half cycles are)
    idx_array = np.asarray([])

    # traverse array, populate idx_array
    for i in range(0, len(input_array)):
        # first and last points
        if (
            i == 0 or
            i == len(input_array) - 1
        ):
            idx_array = np.append(idx_array, i)

        # interior points
        else:
            # detect trough
            if (
                input_array[i] < input_array[i - 1] and
                input_array[i] < input_array[i + 1]
            ):
                idx_array = np.append(idx_array, i)

            # detect peak
            elif (
                input_array[i] > input_array[i - 1] and
                input_array[i] > input_array[i + 1]
            ):
                idx_array = np.append(idx_array, i)

            # detect right plateau
            elif (
                input_array[i] == input_array[i + 1] and
                (
                    input_array[i] < input_array[i - 1] or
                    input_array[i] > input_array[i - 1]
                )
            ):
                idx_array = np.append(idx_array, i)

            # detect left plateau
            elif (
                input_array[i] == input_array[i - 1] and
                (
                    input_array[i] < input_array[i + 1] or
                    input_array[i] > input_array[i + 1]
                )
            ):
                idx_array = np.append(idx_array, i)


    return np.int_(idx_array)


def get_cycle_depths(input_array, idx_array):
    # cast to numpy array
    input_array = np.asarray(input_array)
    idx_array = np.int_(np.asarray(idx_array))

    # init depths array
    depths_array = np.asarray([])

    # traverse half cycles, populate depths array
    for i in range(0, len(idx_array) - 1):
        y_0 = input_array[idx_array[i]]
        y_1 = input_array[idx_array[i + 1]]
        depths_array = np.append(depths_array, abs(y_1 - y_0))

    return depths_array


# ---- models ----


def deshpande_bernardi(
    params,             # array-like of parameters [various units]
    time_array_hrs,     # array-like of time values (time elapsed since 0) [hrs]
    SOC_array           # array-like of SOC values (parallel to time_array_hrs) [ ]
):
    # unpack params
                    # param units
    K = params[0]   # [kWh / hrs^y]
    y = params[1]   # [ ]
    a = params[2]   # [kWh]

    # cast to numpy array
    SOC_array = np.asarray(SOC_array)

    # get time elapsed
    t_hrs = time_array_hrs[-1] - time_array_hrs[0]

    # get half cycles
    idx_array = get_half_cycles(SOC_array)
    depths_array = get_cycle_depths(SOC_array, idx_array)

    # compute DOD sum
    DOD_sum = 0
    for i in range(0, len(depths_array)):
        DOD_sum += math.pow(depths_array[i], 2)

    # compute and return Q_loss
    Q_loss_kWh = K * math.pow(t_hrs, y) + a * DOD_sum
    return Q_loss_kWh  # [kWh]


def quad_approx(
    params,                     # array-like of parameters [various units]
    cap_0_kWh,                  # initial energy capacity [kWh]
    time_array_hrs,             # array-like of time values (time elapsed since 0) [hrs]
    SOC_array,                  # array-like of SOC values (parallel to time_array_hrs) [ ]
    charge_current_array_A,     # array-like of charge current values (parallel to time_array_hrs) [A]
    discharge_current_array_A,  # array-like of discharge current values (parallel to time_array_hrs) [A]
    ambient_temp_avg_K          # average ambient temperature [K]
):
    # cast to numpy array
    SOC_array = np.asarray(SOC_array)
    charge_current_array_A = np.asarray(charge_current_array_A)
    discharge_current_array_A = np.asarray(discharge_current_array_A)

    # get time elapsed
    t_hrs = time_array_hrs[-1] - time_array_hrs[0]

    # get averages
    SOC_avg = np.mean(SOC_array)
    charge_current_avg_A = np.mean(charge_current_array_A)
    discharge_current_avg_A = np.mean(discharge_current_array_A)

    # compute ln_k_c
    ln_k_c = (
                                                                        # param units
        params[0] +                                                     # [ ]
        params[1] * (1 / ambient_temp_avg_K) +                          # [K]
        params[2] * (SOC_avg) +                                         # [ ]
        params[3] * (charge_current_avg_A) +                            # [1 / A]
        params[4] * (discharge_current_avg_A) +                         # [1 / A]
        params[5] * (charge_current_avg_A / ambient_temp_avg_K) +       # [K / A]
        params[6] * (discharge_current_avg_A / ambient_temp_avg_K) +    # [K / A]
        params[7] * (1 / math.pow(ambient_temp_avg_K, 2)) +             # [K^2]
        params[8] * (math.pow(SOC_avg, 2))                              # [ ]
    )

    # compute and return cap_1_kWh
    alpha = params[9]  # [1 / ln(hrs)]
    cap_1_kWh = (
        cap_0_kWh * math.exp(
            -1 * math.exp(
                ln_k_c + alpha * math.log(t_hrs)
            )
        )
    )
    return cap_1_kWh  # [kWh]
