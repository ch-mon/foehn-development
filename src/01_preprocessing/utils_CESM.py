import numpy as np
import dypy.netcdf as dn
import pandas as pd

def read_all_CESM_files(paths, index, lats_labels, lons_labels):
    """
    Read all relevant parameters for all CESM files the CESM file and bring them into tabular format.
    @param paths: List of all paths to read
    @type paths: list
    @param index: Slice of the rectangle within to read coordinates
    @type index: slice
    @param lats_labels: String format of all latitude labels
    @type lats_labels: list
    @param lons_labels: String format of all longitude labels
    @type lons_labels: list
    @return: List of dictionaries with values of the features
    @rtype: list
    """

    # Calculate more constants
    lats_amount = len(lats_labels)
    lons_amount = len(lons_labels)

    # Create all feature names in advance
    feature_names = dict()
    for var in ["T", "V", "U", "Z"]:
        for p in [50000, 70000, 85000]:
            feature_names[f"{var}_{p}"] = [f"{var}_{lat}_{lon}_{p//100}" for lat in lats_labels for lon in lons_labels]
    feature_names["T_90000"] = [f"T_{lat}_{lon}_900" for lat in lats_labels for lon in lons_labels]
    feature_names["SLP"] = [f"SLP_{lat}_{lon}_sealevel" for lat in lats_labels for lon in lons_labels]

    # Loop over all file paths
    rows_list = []
    for filepath in paths:
        print(filepath)

        # Read all raw values
        hyam, hybm = np.array(dn.read_var(filepath, ["hyam", "hybm"]))  # For interpolation to pressure levels
        SLP_values, PS_values = np.array(dn.read_var(filepath, ["PSL", "PS"], index=index))  # PS for interpolation
        T_values, V_values, U_values, Z_values = np.array(dn.read_var(filepath, ["T", "V", "U", "Z3"], index=index))

        # Loop over all time-points in a year (4 time-points/day and always 365 days/year)
        for time_point in range(0, 1460):
            feature_dict = {}

            # Append SLP values
            SLP_values_t = SLP_values[time_point] / 100  # Conversion from Pa to hPa
            feature_dict.update(zip(feature_names["SLP"], SLP_values_t.flatten()))

            # Calculate pressure values at each model level
            P3_t = np.tensordot(hyam, 100000*np.ones((lats_amount, lons_amount)), axes=0) + np.tensordot(hybm, PS_values[time_point], axes=0)

            # Get index of model level which is slightly above 900 hPa
            index_lvl = np.expand_dims((P3_t < 90000).argmin(axis=0), axis=0)
            pressure_lvl_too_low_mask = np.where(index_lvl == 0, True, False)[0]

            # Make linear interpolation for temperature at 900 hPa (only temperature is required on this level)
            p_2 = np.take_along_axis(P3_t, index_lvl, axis=0)[0]  # Pressure value on model level above 900 hPa
            p_1 = np.take_along_axis(P3_t, index_lvl-1, axis=0)[0]  # Pressure value on model level below 900 hPa
            T_2 = np.take_along_axis(T_values[time_point], index_lvl, axis=0)[0]  # Temperature value on model level above 900 hPa
            T_1 = np.take_along_axis(T_values[time_point], index_lvl-1, axis=0)[0]  # Temperature value on model level below 900 hPa
            delta_p = (90000-p_1)/(p_2-p_1)  # Calculate delta p to 900 hPa
            T_interpol = (T_2-T_1)*delta_p + T_1  # Linear interpolation equation

            # If model levels are all above 900 hPa disregard this feature at this grid-point (no extrapolation)
            T_interpol[pressure_lvl_too_low_mask] = np.NaN

            # Append dictionary to row list
            feature_dict.update(zip(feature_names["T_90000"], T_interpol.flatten()))

            # Loop over all relevant pressure levels
            for p in [50000, 70000, 85000]:

                # Get index of model level which is slightly above wanted pressure level
                index_lvl = np.expand_dims((P3_t < p).argmin(axis=0), axis=0)
                pressure_lvl_too_low_mask = np.where(index_lvl == 0, True, False)[0]

                # Get all values at model level above and below the wanted pressure level
                p_2 = np.take_along_axis(P3_t, index_lvl, axis=0)[0]
                p_1 = np.take_along_axis(P3_t, index_lvl-1, axis=0)[0]
                T_2 = np.take_along_axis(T_values[time_point], index_lvl, axis=0)[0]
                T_1 = np.take_along_axis(T_values[time_point], index_lvl-1, axis=0)[0]
                Z_2 = np.take_along_axis(Z_values[time_point], index_lvl, axis=0)[0]
                Z_1 = np.take_along_axis(Z_values[time_point], index_lvl-1, axis=0)[0]
                V_2 = np.take_along_axis(V_values[time_point], index_lvl, axis=0)[0]
                V_1 = np.take_along_axis(V_values[time_point], index_lvl-1, axis=0)[0]
                U_2 = np.take_along_axis(U_values[time_point], index_lvl, axis=0)[0]
                U_1 = np.take_along_axis(U_values[time_point], index_lvl-1, axis=0)[0]

                # Make linear interpolation
                delta_p = (p-p_1)/(p_2-p_1)
                T_interpol = (T_2-T_1)*delta_p + T_1
                V_interpol = (V_2-V_1)*delta_p + V_1
                Z_interpol = (Z_2-Z_1)*delta_p + Z_1
                U_interpol = (U_2-U_1)*delta_p + U_1

                # If model levels are all above the wanted pressure level disregard this feature at this grid-point (no extrapolation)
                T_interpol[pressure_lvl_too_low_mask] = np.NaN
                V_interpol[pressure_lvl_too_low_mask] = np.NaN
                Z_interpol[pressure_lvl_too_low_mask] = np.NaN
                U_interpol[pressure_lvl_too_low_mask] = np.NaN

                # Append dictionary to row list
                feature_dict.update(zip(feature_names[f"T_{p}"], T_interpol.flatten()))
                feature_dict.update(zip(feature_names[f"V_{p}"], V_interpol.flatten()))
                feature_dict.update(zip(feature_names[f"Z_{p}"], Z_interpol.flatten()))
                feature_dict.update(zip(feature_names[f"U_{p}"], U_interpol.flatten()))

            # Append time-point to general list
            rows_list.append(feature_dict)

    return rows_list


def create_date_and_ensemble_columns(years):
    """
    Create date and ensemble member column for a full CESM file.
    @param years: Years for date column
    @type years: list
    @return: Date and ensemble series
    @rtype: pd.Series
    """

    # Construct a date column for all samples in CESM dataframe (each year in CESM has 365 days)
    hours = [0, 6, 12, 18]
    days_in_month = {1: 31,
                     2: 28,
                     3: 31,
                     4: 30,
                     5: 31,
                     6: 30,
                     7: 31,
                     8: 31,
                     9: 30,
                     10: 31,
                     11: 30,
                     12: 31}
    dates = [f"{year}-{month}-{day} {hour}:00" for year in years
                                               for month in range(1, 12 + 1)
                                               for day in range(1, days_in_month[month] + 1)
                                               for hour in hours]

    # Construct an ensemble member list (which repeats entries according to date)
    ensembles = [f"E{nr}" for nr in range(1, 35 + 1) for i in range(len(dates))]

    # Repeat dates 35 times (for all ensemble members)
    dates = dates * 35

    # Create pandas Series
    dates = pd.Series(pd.to_datetime(dates, format="%Y-%m-%d %H:%M"), name="date")
    ensembles = pd.Series(ensembles, name="ensemble")

    # Sanity check
    print("Length of ensemble member column: ", len(ensembles))
    print("Length of date column: ", len(dates))

    return dates, ensembles
