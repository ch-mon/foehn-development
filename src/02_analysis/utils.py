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
    
    # Calculate and append the stability features
    df_CESM = calculate_stability(df = df_CESM, lats = LATS_CESM_STRING, lons = LONS_CESM_STRING)

    # Calculate and append potential temperature features
    df_T= df_CESM.filter(regex=("T\w+900")).add_prefix("PHI")
    df_PHIT_900 = calc_pot_temp(T=df_T, p = 900.0)
    df_T= df_CESM.filter(regex=("T\w+850")).add_prefix("PHI")
    df_PHIT_850 = calc_pot_temp(T=df_T, p = 850.0)
    df_T= df_CESM.filter(regex=("T\w+700")).add_prefix("PHI")
    df_PHIT_700 = calc_pot_temp(T=df_T, p = 700.0)
    
    # Return dataframe with all neceassary basic features
    return pd.concat([df_CESM, df_PHIT_900, df_PHIT_850, df_PHIT_700], axis=1)

def calculate_stability(df, lats, lons):
    '''
    Calculate the stability for 700--900 and 850--900 hPa layers from temperature features for all avaliable grid points.
    '''
    
    stability_dict = {}
    
    for lat in lats:
        for lon in lons:
            try:
                stability_dict[f"DELTAPHI_{lat}_{lon}_700"] = (calc_pot_temp(T=df[f"T_{lat}_{lon}_700"], p = 700.0) - calc_pot_temp(T=df[f"T_{lat}_{lon}_900"], p = 900.0)).values
            except:
                print("Pressure lvl doesnt exist (700-900 hPa, " + lat +", " + lon +")")

            try:
                stability_dict[f"DELTAPHI_{lat}_{lon}_850"] = (calc_pot_temp(T=df[f"T_{lat}_{lon}_850"], p = 850.0) - calc_pot_temp(T=df[f"T_{lat}_{lon}_900"], p = 900.0)).values
            except:
                print("Pressure lvl doesnt exist (850-900 hPa, " + lat +", " + lon +")")
    
    return pd.concat([df, pd.DataFrame.from_dict(stability_dict)], axis=1) # Change this so that it only returns stability dataframe later

def save_figure(name):
    plt.savefig(f'/home/chmony/Documents/Results/newgradient/{name}.pdf', bbox_inches='tight', dpi=200)
    print(f'Saved figure at: /home/chmony/Documents/Results/newgradient/{name}.pdf')
    