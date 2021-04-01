import matplotlib.pyplot as plt
import pandas as pd
import os
# Import constants from constants.py file in directory above
import sys
sys.path.append("..")
from constants import *


def calc_pot_temp(T, p):
    """
    Calculate the potential temperature at a given pressure level.
    @param T: Temperature array (in Kelvin)
    @param p: Pressure array (in hPa)
    """
    p_0 = 1000.0
    c_p = 1005.0
    R = 287.0

    pot_temp = (T)*(p_0/p)**(R/c_p)
    return pot_temp


def load_CESM(filepath, LATS_CESM_STRING, LONS_CESM_STRING):
    """
    Load a before created CESM file (present or future climate), append stability and potential temperature features and return full dataframe.
    @param filepath: filepath to CESMp or CESMf file
    @param LATS_CESM_STRING: CESM latitude coordinates
    @param LONS_CESM_STRING: CESM longitude coordinates
    @return: Preprocessed CESMp or CESMf file
    """
    df_CESM = pd.read_csv(filepath)
    df_CESM["date"] = pd.to_datetime(df_CESM["date"], format="%Y-%m-%dT%H:%M:00.000000Z")  # Due to Dataiku date format, evtl adjust this
    
    # Calculate and append potential temperature features
    df_T= df_CESM.filter(regex=("T\w+900")).add_prefix("PHI")
    df_PHIT_900 = calc_pot_temp(T=df_T, p=900.0)
    df_T= df_CESM.filter(regex=("T\w+850")).add_prefix("PHI")
    df_PHIT_850 = calc_pot_temp(T=df_T, p=850.0)
    df_T= df_CESM.filter(regex=("T\w+700")).add_prefix("PHI")
    df_PHIT_700 = calc_pot_temp(T=df_T, p=700.0)
    
    df_CESM = pd.concat([df_CESM, df_PHIT_900, df_PHIT_850, df_PHIT_700], axis=1)
    
    # Calculate and append the stability features
    df_CESM_stability = calculate_stability(df=df_CESM, lats=LATS_CESM_STRING, lons=LONS_CESM_STRING)
    df_CESM = pd.concat([df_CESM, df_CESM_stability], axis=1)

    # Return dataframe with all necessary basic features
    return df_CESM


def calculate_stability(df, lats, lons):
    """
    Calculate the stability for 700--900 and 850--900 hPa layers from temperature features for all available grid points in lats and lons.
    @param df: ERAI or CESM dataframe
    @param lats: CESM latitude coordinates
    @param lons: CESM longitude coordinates
    @return: Stability variable dataframe
    """
    
    stability_dict = {}
    # Loop over all coordinates
    for lat in lats:
        for lon in lons:
            for level in ["700", "850"]:
                try:
                    stability_dict[f"DELTAPHI_{lat}_{lon}_{level}"] = (df[f"PHIT_{lat}_{lon}_{level}"] - df[f"PHIT_{lat}_{lon}_900"]).values
                except:
                    # Might happen since not all features are available in CESM
                    pass
    
    return pd.DataFrame(stability_dict) 


def save_figure(name):
    """
    Save a PDF figure to figures folder.
    @param name: Name of the figure
    """
    filepath = os.path.join(BASE_DIR, f'figures/{name}.pdf')
    plt.savefig(filepath, bbox_inches='tight', dpi=200)
    print(f'Saved figure at: {filepath}')
    
    
def calculate_horizontal_feature_differences(df, variable, pressure_levels):
    """
    Calculate the horizontal potential temperature differences for a given variable and pressure levels.
    @param df: ERAI or CESM dataframe
    @param variable: Variable for which to calculate the horizontal difference
    @param pressure_levels: For which pressure levels to calculate the horizontal differences
    @return: Dataframe with horizontal differences
    """
    
    df_variable = df.filter(regex=(f"{variable}_\w+"))
    
    feature_dict = {}
    for level in pressure_levels:
        df_level = df_variable.filter(regex=(f"\w+_{level}"))

        variable_list1 = sorted(df_level.columns.tolist())
        variable_list2 = sorted(df_level.columns.tolist())
        
        for col1 in variable_list1:
            variable_list2.remove(col1)
            for col2 in variable_list2:
                feature_dict[f"diff_{col1}_{col2}"] = (df_level.loc[:, col1] - df_level.loc[:, col2]).values

    return pd.DataFrame(feature_dict)


def generate_reduced_features_on_CESM(feature_to_generate, df_CESM):
    """
    Generate the most important ERAI features on CESM.
    @param feature_to_generate: List of most important features
    @type feature_to_generate: list
    @param df_CESM: CESM dataframe (present or future)
    @type df_CESM: pd.DataFrame
    @return: Dataframe with most important features, date and ensemble member
    @rtype: pd.DataFrame
    """

    # Loop over all features
    feature_dict_CESM = {}
    for feature_name in feature_to_generate:
        if feature_name[0:2] == "V_" or feature_name[0:2] == "U_":  # Wind features
            feature_dict_CESM[feature_name] = df_CESM.loc[:, feature_name].values
        elif feature_name[0:6] == "DELTAP":  # Stability features
            feature_name_splitted = feature_name.split("_")
            first_feature = "PHIT_" + "_".join(feature_name_splitted[1:4])
            second_feature = "PHIT_" + "_".join(feature_name_splitted[1:3]) + "_900"
            feature_dict_CESM["DELTAPHI_" + "_".join(feature_name_splitted[1:4])] = (df_CESM.loc[:, first_feature] - df_CESM.loc[:, second_feature]).values
        else:  # SLP, Z, and PHIT features
            feature_name_splitted = feature_name.split("_")
            first_feature = "_".join(feature_name_splitted[1:5])
            second_feature = "_".join(feature_name_splitted[5:9])
            feature_dict_CESM[f"diff_{first_feature}_{second_feature}"] = (df_CESM.loc[:, first_feature] - df_CESM.loc[:, second_feature]).values

    # Append data and ensemble member feature
    feature_dict_CESM["date"] = df_CESM.loc[:, "date"].values
    feature_dict_CESM["ensemble"] = df_CESM.loc[:, "ensemble"].values

    return pd.DataFrame(feature_dict_CESM)