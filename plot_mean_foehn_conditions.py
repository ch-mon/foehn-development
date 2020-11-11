import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dypy.plotting import Mapfigure
import matplotlib

from utils import calc_pot_temp

def calculate_stability(df, lats, lons):
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
    
    return pd.concat([df, pd.DataFrame.from_dict(stability_dict)], axis=1)

def transform_to_2D_grid(df, variable, variable_lvl, lats_labels, lons_labels):
    grid_foehn = np.zeros((len(lats_labels), len(lons_labels)))
    
    if variable == "U":
        grid2_foehn = np.zeros((len(lats_labels), len(lons_labels)))
        for index_lat, lat in enumerate(lats_labels):
            for index_lon, lon in enumerate(lons_labels):
                grid_foehn[index_lat][index_lon] = df.loc[:,f"{variable}_{lat}_{lon}_{variable_lvl}"].mean()
                grid2_foehn[index_lat][index_lon] = df.loc[:,f"V_{lat}_{lon}_{variable_lvl}"].mean()
                
        return grid_foehn, grid2_foehn
    
    else:
        for index_lat, lat in enumerate(lats_labels):
            for index_lon, lon in enumerate(lons_labels):
                try:
                    grid_foehn[index_lat][index_lon] = df.loc[:,f"{variable}_{lat}_{lon}_{variable_lvl}"].mean()
                except:
                    grid_foehn[index_lat][index_lon] = np.NaN
        
        return grid_foehn
    
def create_vectorfield(grid_U, grid_V, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels):
  
    lats = [int(lat)/100.0 for lat in lats_labels]
    lons = [int(lon)/100.0 for lon in lons_labels]

    mf = Mapfigure(lon=np.array(lons), lat=np.array(lats))
    fig = plt.figure(figsize=(16,9))

    grid_ges = np.sqrt(grid_U**2+grid_V**2)

    norm = matplotlib.colors.Normalize()
    norm.autoscale(grid_ges.flatten())
    cm = plt.cm.get_cmap('rocket_r', 20)

    sm = plt.cm.ScalarMappable(cmap=cm)
    sm.set_array(grid_ges.flatten())

    qui = plt.quiver(lons, lats, grid_U, grid_V, scale_units="xy", color=cm(norm(grid_ges.flatten())))
    
    cbar = plt.colorbar(sm,
                        boundaries=np.round(np.linspace(vmin, vmax, 19),1),
                        ticks=np.round(np.linspace(vmin, vmax, 19),1),
                        extend="both")
    
    cbar.set_label(unit, rotation=90, labelpad=10, fontsize=14)
    
#    pressure_difference_amount = 31
#    df_sample = df_importances.loc[0:pressure_difference_amount-1,:]
#    for i in range(len(df_sample.lat1)):
#        plt.plot(df_sample.loc[i, "lon1"]*0.01,
#                     df_sample.loc[i, "lat1"]*0.01,
#                     "bx",
#                     markersize=14*(len(df_sample.lat1)-i)/len(df_sample.lat1),
#                     alpha=(len(df_sample.lat1)-i)/len(df_sample.lat1),
#                     mew=4)

#     plt.plot(df_sample.loc[len(df_sample.lat1)-1, "lon1"]*0.01,
#                      df_sample.loc[len(df_sample.lat1)-1, "lat1"]*0.01,
#                      "rx",
#                      markersize=14,
#                      alpha=1,
#                      mew=4)

    plt.plot(8.64441, 46.88042, 'o', color="#00FF00",markersize=8)
    mf.drawmap(nbrem=1, nbrep=1)

    plt.savefig(f'/home/chmony/Documents/Results/newgradient/weathermap_{variable}_{variable_lvl}_{model}.pdf', bbox_inches='tight', dpi=200)
    print(f"Saved figure at: /home/chmony/Documents/Results/newgradient/weathermap_{variable}_{variable_lvl}_{model}.pdf'")

# Obtain ERA-Interim mean grid configurations
def create_contour(grid, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels):
  
    lats = [int(lat)/100.0 for lat in lats_labels]
    lons = [int(lon)/100.0 for lon in lons_labels]

    mf = Mapfigure(lon=np.array(lons), lat=np.array(lats))
    fig = plt.figure(figsize=(16,9))
    plt.title(f"{variable} at {variable_lvl} hPa for {model} data")
    
    cnt = plt.contourf(lons, lats, grid, 20, cmap=plt.cm.get_cmap('rocket'), vmin=vmin, vmax=vmax)
    m = plt.cm.ScalarMappable(cmap=plt.cm.get_cmap('rocket', 20))
    m.set_array(grid)
    m.set_clim(vmin, vmax)
 
    cbar = plt.colorbar(m,
                        boundaries=np.round(np.linspace(vmin, vmax, 19),1),
                        ticks=np.round(np.linspace(vmin, vmax, 19),1),
                        extend="both")
    cbar.set_label(unit, rotation=90, labelpad=10, fontsize=14)

#     for c in cnt.collections:
#         c.set_edgecolor("face")

#    pressure_difference_amount = 40
#    df_sample = importances.loc[0:pressure_difference_amount-1,:]
#    if variable == "DELTAPHI":
#        for i in range(len(df_sample.lat1)):
#            plt.plot(df_sample.loc[i, "lon1"]*0.01,
#                     df_sample.loc[i, "lat1"]*0.01,
#                     "bx",
#                     markersize=12*(len(df_sample.lat1)-i)/len(df_sample.lat1),
#                     alpha=(len(df_sample.lat1)-i)/len(df_sample.lat1),
#                     mew=4)
#
#        end = len(df_sample.lat1)-1
#        print(df_sample.loc[end,:])
#        plt.plot(df_sample.loc[end-1, "lon1"]*0.01,
#                     df_sample.loc[end-1, "lat1"]*0.01,
#                     "rx",
#                     markersize=12,
#                     alpha=1,
#                     mew=4)
#        plt.plot(df_sample.loc[end, "lon1"]*0.01,
#                     df_sample.loc[end, "lat1"]*0.01,
#                     "rx",
#                     markersize=12,
#                     alpha=1,
#                     mew=4)
#    else:
#        for i in range(len(df_sample.lat1)):
#            plt.plot([df_sample.loc[i, "lon1"]*0.01, df_sample.loc[i, "lon2"]*0.01],
#                    [df_sample.loc[i, "lat1"]*0.01, df_sample.loc[i, "lat2"]*0.01],
#                    alpha=(len(df_sample.lat1)-i)/len(df_sample.lat1),
#                    c='b',
#                    linewidth=5*(len(df_sample.lat1)-i)/len(df_sample.lat1))
#
#        end = len(df_sample.lat1)-1
#        print(df_sample.loc[end,:])
#        plt.plot([df_sample.loc[end, "lon1"]*0.01, df_sample.loc[end, "lon2"]*0.01],
#                    [df_sample.loc[end, "lat1"]*0.01, df_sample.loc[end, "lat2"]*0.01],
#                    alpha=1,
#                    c='r',
#                    linewidth=5)
    plt.plot(8.6833, 46.5167, 'o', color="#00FF00",markersize=8) #Altdorf 8.64441, 46.88042
    mf.drawmap(nbrem=1, nbrep=1)

    plt.savefig(f'/home/chmony/Documents/Results/newgradient/weathermap_{variable}_{variable_lvl}_{model}.pdf', bbox_inches='tight', dpi=200)
    print(f"Saved figure at: /home/chmony/Documents/Results/newgradient/weathermap_{variable}_{variable_lvl}_{model}.pdf'")

    
def plot_mean_foehn_condition_for_one_model(variable, variable_lvl, unit, model, vmin, vmax, df, foehn, lats_labels, lons_labels):
    df_foehn = df.loc[foehn == 1, :]
    
    if variable == "U":
        grid_U, grid_V = transform_to_2D_grid(df_foehn, variable, variable_lvl, lats_labels, lons_labels)
        create_vectorfield(grid_U, grid_V, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels)
    else:
        grid = transform_to_2D_grid(df_foehn, variable, variable_lvl, lats_labels, lons_labels)
        create_contour(grid, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels)
        
    


def execute_for_all_dataframes(variable, variable_lvl, df_ERA, df_CESMp, df_CESMf, foehn_obs, foehn_ERA, foehn_CESMp, foehn_CESMf, lats_labels, lons_labels):
    df_obs = df_ERA.loc[foehn_obs == 1, :]
    df_ERA = df_ERA.loc[foehn_ERA == 1, :]
    df_CESMp = df_CESMp.loc[foehn_CESMp == 1, :]
    df_CESMf = df_CESMf.loc[foehn_CESMf == 1, :]
    
    
    transform_to_2D_grid(df_ERA)
    transform_to_2D_grid(df_ERA)
    
if __name__ == "__main__":
    print("Hello World.")
    