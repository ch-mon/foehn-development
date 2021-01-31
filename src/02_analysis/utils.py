import matplotlib.pyplot as plt
import pandas as pd

def calc_pot_temp(T, p):
    '''
    Calculate the potential temperature at a given pressure level.
    '''
    p_0 = 1000.0
    c_p = 1005.0
    R = 287.0

    pot_temp = (T)*(p_0/p)**(R/c_p)
    return pot_temp

def load_CESM(filepath, LATS_CESM_STRING, LONS_CESM_STRING):
    '''
    Load a before created CESM file (present or future climate), append stability and potential temperature features and return full dataframe.
    '''
    df_CESM = pd.read_csv(filepath)
    df_CESM["date"] = pd.to_datetime(df_CESM["date"], format="%Y-%m-%dT%H:%M:00.000000Z") # Due to Dataiku date format, evtl adjust this
    
    # Calculate and append potential temperature features
    df_T= df_CESM.filter(regex=("T\w+900")).add_prefix("PHI")
    df_PHIT_900 = calc_pot_temp(T=df_T, p=900.0)
    df_T= df_CESM.filter(regex=("T\w+850")).add_prefix("PHI")
    df_PHIT_850 = calc_pot_temp(T=df_T, p=850.0)
    df_T= df_CESM.filter(regex=("T\w+700")).add_prefix("PHI")
    df_PHIT_700 = calc_pot_temp(T=df_T, p=700.0)
    
    df_CESM = pd.concat([df_CESM, df_PHIT_900, df_PHIT_850, df_PHIT_700], axis=1)
    
    # Calculate and append the stability features
    df_CESM_stability = calculate_stability(df = df_CESM, lats = LATS_CESM_STRING, lons = LONS_CESM_STRING)
    df_CESM = pd.concat([df_CESM, df_CESM_stability], axis=1)

    # Return dataframe with all neceassary basic features
    return df_CESM


def calculate_stability(df, lats, lons):
    '''
    Calculate the stability for 700--900 and 850--900 hPa layers from temperature features for all avaliable grid points.
    '''
    
    stability_dict = {}
    
    for lat in lats:
        for lon in lons:
            try:
                stability_dict[f"DELTAPHI_{lat}_{lon}_700"] = (df[f"PHIT_{lat}_{lon}_700"] - df[f"PHIT_{lat}_{lon}_900"]).values
            except:
                # Might happen since not all features are avaliable in CESM
                pass

            try:
                stability_dict[f"DELTAPHI_{lat}_{lon}_850"] = (df[f"PHIT_{lat}_{lon}_850"] - df[f"PHIT_{lat}_{lon}_900"]).values
            except:
                # Might happen since not all features are avaliable in CESM
                pass
    
    return pd.DataFrame(stability_dict) 

def save_figure(name):
    plt.savefig(f'/home/chmony/Documents/Results/newgradient/{name}.pdf', bbox_inches='tight', dpi=200)
    print(f'Saved figure at: /home/chmony/Documents/Results/newgradient/{name}.pdf')
    
    
def calculate_horizontal_feature_differences(df, variable, pressure_levels):
    
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
