def calc_pot_temp(T, p):
    p_0 = 1000.0
    c_p = 1005.0
    R = 287.0

    pot_temp = (T)*(p_0/p)**(R/c_p)
    return pot_temp

def load_CESM(filepath, LATS_CESM_STRING, LONS_CESM_STRING):
    '''
    Load a before created CESM file (present or future climate), append stability and pot. temp features and return full dataframe.
    '''
    df_CESM = pd.read_csv(filepath)
    df_CESM["date"] = pd.to_datetime(df_CESM["date"], format="%Y-%m-%dT%H:%M:00.000000Z") # Due to Dataiku date format
    
    # Calculate and append the stability features
    df_CESM = calculate_stability(df = df_CESM, lats = LATS_CESM_STRING, lons = LONS_CESM_STRING)

    # Calculate and append pot. temp features
    df_T= df_CESM.filter(regex=("T\w+900")).add_prefix("PHI")
    df_PHIT_900 = calc_pot_temp(T=df_T, p = 900.0)
    df_T= df_CESM.filter(regex=("T\w+850")).add_prefix("PHI")
    df_PHIT_850 = calc_pot_temp(T=df_T, p = 850.0)
    df_T= df_CESM.filter(regex=("T\w+700")).add_prefix("PHI")
    df_PHIT_700 = calc_pot_temp(T=df_T, p = 700.0)
    
    # Return dataframe with all neceassary basic features
    return pd.concat([df_CESM, df_PHIT_900, df_PHIT_850, df_PHIT_700], axis=1)