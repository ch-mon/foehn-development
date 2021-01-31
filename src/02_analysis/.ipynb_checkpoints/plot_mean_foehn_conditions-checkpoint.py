import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dypy.plotting import Mapfigure
import matplotlib

from utils import calc_pot_temp, calculate_stability

def transform_to_2D_grid(df, variable, variable_lvl, lats_labels, lons_labels):
    '''
    Transform dataframes into 2D arrays.
    '''
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
    
def create_vectorfield(grid_U, grid_V, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels, df_importances):
    '''
    Create vector field plot for U and V variables.
    '''
  
    lats = [int(lat)/100.0 for lat in lats_labels]
    lons = [int(lon)/100.0 for lon in lons_labels]

    mf = Mapfigure(lon=np.array(lons), lat=np.array(lats))
    fig = plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 20})

    grid_ges = np.sqrt(grid_U**2+grid_V**2)

    norm = matplotlib.colors.Normalize()
    norm.autoscale(grid_ges.flatten())
    cm = plt.cm.get_cmap('rocket_r')

    sm = plt.cm.ScalarMappable(cmap=cm)
    sm.set_array(grid_ges.flatten())

    qui = plt.quiver(lons, lats, grid_U, grid_V, scale_units="xy", color=cm(norm(grid_ges.flatten())))
    
    levels= np.arange(vmin, vmax, 1)
    ticks= levels
    cbar = plt.colorbar(sm,
                        boundaries=levels,
                        ticks=ticks,
                        extend="both", fraction=0.0188, pad=0.03)
    
    cbar.set_label(unit, rotation=90, labelpad=10)

    for i in df_importances.index:
        plt.plot(df_importances.loc[i, "lon1"]*0.01,
                    df_importances.loc[i, "lat1"]*0.01,
                    "bx",
                    markersize=12,
                    alpha=df_importances.loc[i, "importance"]/df_importances["importance"].max(),
                    mew=4)

    mf.drawmap(nbrem=1, nbrep=1)
    plt.plot(8.96004, 46.01008, 'o', color="#00FF00",markersize=8)
    

    plt.savefig(f'/home/chmony/Documents/Results/newgradient/weathermap_{variable}_{variable_lvl}_{model}.pdf', bbox_inches='tight', dpi=200)
    print(f"Saved figure at: /home/chmony/Documents/Results/newgradient/weathermap_{variable}_{variable_lvl}_{model}.pdf'")

def generate_coordinates_from_feature_label(df_features, variable):
    '''
    Extract coordinates from feauture names again. Necessary for plotting most important differences.
    '''
    
    feature_list = [feature.split("_") for feature in df_features.keys() if variable + "_" in feature]
    importance_list = [value for feature, value in df_features.items() if variable + "_" in feature]
    if variable=="U":
        feature_list += [feature.split("_") for feature in df_features.keys() if "V_" in feature]
        importance_list += [value for feature, value in df_features.items() if "V_" in feature]
        coordinates = [[feature[0], feature[1], feature[2]] for feature in feature_list]
        df_importances = pd.DataFrame(coordinates, columns =["variable", "lat1", "lon1"])
        df_importances[["lat1", "lon1"]] = df_importances[["lat1", "lon1"]].astype(int)
        df_importances["importance"] = importance_list
        return df_importances
    elif variable=="DELTAPHI":
        coordinates = [[feature[0], feature[1], feature[2]] for feature in feature_list]
        df_importances = pd.DataFrame(coordinates, columns =["variable", "lat1", "lon1"])
        df_importances[["lat1", "lon1"]] = df_importances[["lat1", "lon1"]].astype(int)
        df_importances["importance"] = importance_list
        return df_importances
    else:
        coordinates = [[feature[1], feature[2], feature[3], feature[6], feature[7]] for feature in feature_list]
        df_importances = pd.DataFrame(coordinates, columns =["variable", "lat1", "lon1", "lat2", "lon2"], dtype=int)
        df_importances[["lat1", "lon1", "lat2", "lon2"]] = df_importances[["lat1", "lon1", "lat2", "lon2"]].astype(int)
        df_importances["importance"] = importance_list
        return df_importances
    
# Obtain ERA-Interim mean grid configurations
def create_contour(grid, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels, df_importances, location):
    '''
    Create contour plot for a specified variable at a given pressure level.
    '''
  
    lats = [int(lat)/100.0 for lat in lats_labels]
    lons = [int(lon)/100.0 for lon in lons_labels]

    mf = Mapfigure(lon=np.array(lons), lat=np.array(lats))
#     fig = plt.figure(figsize=(16,9))
    plt.rcParams.update({'font.size': 20})
#     plt.title(f"{variable} at {variable_lvl} hPa for {model} data")
    
    if vmax-vmin<=10.2:
        levels = np.arange(vmin, vmax, 0.5)
        ticks= levels[1::2]
    elif vmax-vmin<=21:
        levels= np.arange(vmin, vmax, 1)
        ticks= levels[1::2]
    else:
        levels= np.arange(vmin, vmax, 10)
        ticks= levels[1::2]
        
    cmap=plt.cm.get_cmap('rocket')
    
    cnt = plt.contourf(lons, lats, grid, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)
#     cnt_l =  plt.contour(lons, lats, grid, levels=np.round(np.linspace(vmin, vmax, 19),1), colors="k", linewidths=0.1, vmin=vmin, vmax=vmax)
#     plt.clabel(cnt_l, inline=0, fontsize=18, fmt="%1.1f",  levels= cnt_l.levels[1:-1:2])
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(grid)
    m.set_clim(vmin, vmax)
 
    cbar = plt.colorbar(m,
                        boundaries=levels,
                        ticks=ticks,
                        extend="both", fraction=0.0188, pad=0.03)
    cbar.set_label(unit, rotation=90, labelpad=10)

    for c in cnt.collections:
        c.set_edgecolor("face")


    if variable == "DELTAPHI":
        for i in range(len(df_importances.index)):
            plt.plot(df_importances.loc[i, "lon1"]*0.01,
                    df_importances.loc[i, "lat1"]*0.01,
                    "bx",
                    markersize=12,
                    alpha=df_importances.loc[i, "importance"]/df_importances["importance"].max(),
                    mew=4)

    else:
        for i in range(len(df_importances.index)):
            plt.plot([df_importances.loc[i, "lon1"]*0.01, df_importances.loc[i, "lon2"]*0.01],
                   [df_importances.loc[i, "lat1"]*0.01, df_importances.loc[i, "lat2"]*0.01],
                   alpha=df_importances.loc[i, "importance"]/df_importances["importance"].max(),
                   c='b',
                   linewidth=5)

    
    mf.drawmap(nbrem=1, nbrep=1)
    #Plot Foehn location (Altdorf 8.64441, 46.88042, Piotta: 8.6833, 46.5167, Lugano: 8.96004, 46.01008)
    if location == "ALT":
        plt.plot(8.64441, 46.88042, 'o', color="#00FF00",markersize=8)
    elif location == "LUG":
        plt.plot(8.96004, 46.01008, 'o', color="#00FF00",markersize=8)
        
   
    #plt.savefig(f'/home/chmony/Documents/Results/newgradient/weathermap_{variable}_{variable_lvl}_{model}.pdf', bbox_inches='tight', dpi=200)
    #print(f"Saved figure at: /home/chmony/Documents/Results/newgradient/weathermap_{variable}_{variable_lvl}_{model}.pdf'")

    
def plot_mean_foehn_condition_for_one_model(variable, variable_lvl, unit, model, vmin, vmax, df, foehn, lats_labels, lons_labels, df_importances,location):
    '''
    Plot composite map for a given variable for all positive foehn instances.
    '''
    df_foehn = df.loc[foehn == 1, :]
    
    if variable == "U":
        grid_U, grid_V = transform_to_2D_grid(df_foehn, variable, variable_lvl, lats_labels, lons_labels)
        create_vectorfield(grid_U, grid_V, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels, df_importances, location)
    else:
        grid = transform_to_2D_grid(df_foehn, variable, variable_lvl, lats_labels, lons_labels)
        create_contour(grid, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels, df_importances, location)
        
    
if __name__ == "__main__":
    pass
    