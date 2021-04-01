import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dypy.plotting import Mapfigure
import matplotlib


def generate_coordinates_from_feature_label(df_features, variable):
    """
    Extract coordinates from feature names again. Necessary for plotting most important differences.
    @param df_features: Dataframe with most important features
    @param variable: Variable to be plotted (e.g., SLP)
    """

    # Create a list with splitted feature names and another one with the respective importances
    feature_list = [feature.split("_") for feature in df_features.keys() if variable + "_" in feature]
    importance_list = [value for feature, value in df_features.items() if variable + "_" in feature]

    if variable == "U":  # For wind variables
        # Add V variables here as well
        feature_list += [feature.split("_") for feature in df_features.keys() if "V_" in feature]
        importance_list += [value for feature, value in df_features.items() if "V_" in feature]

        # Create dataframe with variable name, latitude, and longitude
        df_importances = pd.DataFrame([[feature[0], feature[1], feature[2]] for feature in feature_list], columns=["variable", "lat1", "lon1"])
        df_importances[["lat1", "lon1"]] = df_importances[["lat1", "lon1"]].astype(int)/100

    elif variable == "DELTAPHI":  # For stability variables
        # Create dataframe with variable name, latitude, and longitude
        df_importances = pd.DataFrame([[feature[0], feature[1], feature[2]] for feature in feature_list], columns=["variable", "lat1", "lon1"])
        df_importances[["lat1", "lon1"]] = df_importances[["lat1", "lon1"]].astype(int)/100

    else:  # SLP, Z, and T
        # Create dataframe with variable name, latitude1, longitude1, latitude2, and longitude2
        df_importances = pd.DataFrame([[feature[1], feature[2], feature[3], feature[6], feature[7]] for feature in feature_list], columns=["variable", "lat1", "lon1", "lat2", "lon2"], dtype=int)
        df_importances[["lat1", "lon1", "lat2", "lon2"]] = df_importances[["lat1", "lon1", "lat2", "lon2"]].astype(int)/100

    # Add importances
    df_importances["importance"] = importance_list
    df_importances = df_importances.sort_values(by="importance", ascending=False)
    return df_importances


def transform_to_2D_grid(df, variable, variable_lvl, lats_labels, lons_labels):
    """
    Transform dataframes into 2D arrays of means at each grid coordinate
    @param df: ERAI or CESM dataframe
    @param variable: Variable to transform
    @param variable_lvl: Pressure level for which to perform the transformation
    @param lats_labels: CESM latitudes
    @param lats_labels: CESM longitudes
    @return 2D numpy array with mean values
    """

    # Initialize empty grid
    grid_foehn = np.zeros((len(lats_labels), len(lons_labels)))

    # For wind variables treat U and V separately
    if variable == "U":
        grid2_foehn = np.zeros((len(lats_labels), len(lons_labels)))

        # Loop over all grid coordinates
        for index_lat, lat in enumerate(lats_labels):
            for index_lon, lon in enumerate(lons_labels):
                grid_foehn[index_lat][index_lon] = df.loc[:, f"{variable}_{lat}_{lon}_{variable_lvl}"].mean()
                grid2_foehn[index_lat][index_lon] = df.loc[:, f"V_{lat}_{lon}_{variable_lvl}"].mean()
                
        return grid_foehn, grid2_foehn
    
    else:  # All other variables
        # Loop over all grid coordinates
        for index_lat, lat in enumerate(lats_labels):
            for index_lon, lon in enumerate(lons_labels):
                try:
                    grid_foehn[index_lat][index_lon] = df.loc[:, f"{variable}_{lat}_{lon}_{variable_lvl}"].mean()
                except:  # If variable does not exist at this grid point
                    grid_foehn[index_lat][index_lon] = np.NaN
        
        return grid_foehn


def create_contour(grid, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels, df_importances, location):
    """
    Create contour plot for a specified variable at a given pressure level.
    @param grid: 2D grid with mean values to plot as contour
    @param variable: Variable of interest
    @param variable_lvl: Level at which variable is plotted (for title)
    @param unit: Unit of variable
    @param model: Which data is used for plotting (for title)
    @param vmin: Lower bound for colorbar
    @param vmax: Upper bound for colorbar
    @param lats_labels: CESM latitudes
    @param lons_labels: CESM longitudes
    @param df_importances: Dataframe with variable importances
    @param location: Location of foehn station
    """

    # Convert string labels to floats
    lats = [int(lat) / 100.0 for lat in lats_labels]
    lons = [int(lon) / 100.0 for lon in lons_labels]

    # Create mapfigure and set title (if wanted)
    mf = Mapfigure(lon=np.array(lons), lat=np.array(lats))
    mf.drawmap(nbrem=1, nbrep=1)
    # plt.title(f"{variable} at {variable_lvl} hPa for {model} data")

    # Set correct colorbar intervals in plot
    if vmax - vmin <= 10.2:
        levels = np.arange(vmin, vmax, 0.5)
        ticks = levels[1::2]
    elif vmax - vmin <= 21:
        levels = np.arange(vmin, vmax, 1)
        ticks = levels[1::2]
    else:
        levels = np.arange(vmin, vmax, 10)
        ticks = levels[1::2]

    # Make contour plot
    cmap = plt.cm.get_cmap('rocket')
    cnt = plt.contourf(lons, lats, grid, levels=levels, cmap=cmap, vmin=vmin, vmax=vmax)

    # If no white lines between levels in contour plot are wanted, uncomment this
    # for c in cnt.collections:
    #     c.set_edgecolor("face")

    # Format colorbar nicely
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(grid)
    m.set_clim(vmin, vmax)
    cbar = plt.colorbar(m,
                        boundaries=levels,
                        ticks=ticks,
                        extend="both", fraction=0.0188, pad=0.03)
    cbar.set_label(unit, rotation=90, labelpad=10)

    # Plot the feature importances on the contour plot
    if variable == "DELTAPHI":  # In case of stability, plot a cross
        for i in df_importances.index:
            plt.plot(df_importances.loc[i, "lon1"] * 0.01,
                     df_importances.loc[i, "lat1"] * 0.01,
                     "bx",
                     markersize=12 * df_importances.loc[i, "importance"] / df_importances["importance"].max(),
                     alpha=df_importances.loc[i, "importance"] / df_importances["importance"].max(),
                     mew=4)

    else:  # Plot lines in all other cases
        for i in df_importances.index:
            plt.plot([df_importances.loc[i, "lon1"], df_importances.loc[i, "lon2"]],
                     [df_importances.loc[i, "lat1"], df_importances.loc[i, "lat2"]],
                     linewidth=5 * df_importances.loc[i, "importance"] / df_importances["importance"].max(),
                     c='b')

    # Plot station location (Altdorf 8.64441, 46.88042, Lugano: 8.96004, 46.01008)
    if location == "ALT":
        plt.plot(8.64441, 46.88042, 'o', color="#00FF00", markersize=8)
    elif location == "LUG":
        plt.plot(8.96004, 46.01008, 'o', color="#00FF00", markersize=8)

    # Uncomment if you want to save the figure alone
    # plt.savefig(f'weathermap_{variable}_{variable_lvl}_{model}.pdf', bbox_inches='tight', dpi=200)
    # print(f"Saved figure at: weathermap_{variable}_{variable_lvl}_{model}.pdf")


def create_vectorfield(grid_U, grid_V, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels, df_importances, location):
    """
    Create vector field plot for U and V variables.
    @param grid_U: 2D grid with mean U values
    @param grid_V: 2D grid with mean V values
    @param variable: Variable of interest
    @param variable_lvl: Level at which variable is plotted (for title)
    @param unit: Unit of variable
    @param model: Which data is used for plotting (for title)
    @param vmin: Lower bound for colorbar
    @param vmax: Upper bound for colorbar
    @param lats_labels: CESM latitudes
    @param lons_labels: CESM longitudes
    @param df_importances: Dataframe with variable importances
    @param location: Location of foehn station
    """

    # Convert string labels to floats
    lats = [int(lat)/100.0 for lat in lats_labels]
    lons = [int(lon)/100.0 for lon in lons_labels]

    # Create mapfigure and set title (if wanted)
    mf = Mapfigure(lon=np.array(lons), lat=np.array(lats))
    mf.drawmap(nbrem=1, nbrep=1)
    # plt.title(f"{variable} at {variable_lvl} hPa for {model} data")

    # Calculate norm of U and V vectors
    grid_ges = np.sqrt(grid_U**2+grid_V**2)

    # Set correct colorbar intervals in plot
    norm = matplotlib.colors.Normalize()
    norm.autoscale(grid_ges.flatten())
    cm = plt.cm.get_cmap('rocket_r')
    sm = plt.cm.ScalarMappable(cmap=cm)
    sm.set_array(grid_ges.flatten())
    
    # Make vector plot
    plt.quiver(lons, lats, grid_U, grid_V, scale_units="xy", color=cm(norm(grid_ges.flatten())))
    
    # Format colorbar nicely
    ticks = np.arange(vmin, vmax, 1)
    cbar = plt.colorbar(sm,
                        boundaries=ticks,
                        ticks=ticks,
                        extend="both", fraction=0.0188, pad=0.03)
    cbar.set_label(unit, rotation=90, labelpad=10)

    # Plot the feature importances on the contour plot
    for i in df_importances.index:
        plt.plot(df_importances.loc[i, "lon1"],
                 df_importances.loc[i, "lat1"],
                 "bx",
                 markersize=12*df_importances.loc[i, "importance"]/df_importances["importance"].max(),
                 mew=4)

    
    
    # Plot station location (Altdorf 8.64441, 46.88042, Lugano: 8.96004, 46.01008)
    if location == "ALT":
        plt.plot(8.64441, 46.88042, 'o', color="#00FF00",markersize=8)
    elif location == "LUG":
        plt.plot(8.96004, 46.01008, 'o', color="#00FF00",markersize=8)
    
    # Uncomment if you want to save the figure solely
    # plt.savefig(f'weathermap_{variable}_{variable_lvl}_{model}.pdf', bbox_inches='tight', dpi=200)
    # print(f"Saved figure at: weathermap_{variable}_{variable_lvl}_{model}.pdf")


def plot_mean_foehn_condition_for_one_model(variable, variable_lvl, unit, model, vmin, vmax, df, foehn, lats_labels, lons_labels, df_importances, location):
    """
    Plot composite map for a given variable for all positive foehn instances for ERAI, CESMp or CESMf data
    @param variable: Variable of interest
    @param variable_lvl: Level at which variable is plotted (for title)
    @param unit: Unit of variable
    @param model: Which data is used for plotting (for title)
    @param vmin: Lower bound for colorbar
    @param vmax: Upper bound for colorbar
    @param df: Dataframe with ERAI or CESM data
    @param foehn: Series with Foehn index
    @param lats_labels: CESM latitudes
    @param lons_labels: CESM longitudes
    @param df_importances: Dataframe with variable importances
    @param location: Location of foehn station
    """

    # Select where foehn is prevalent
    df_foehn = df.loc[foehn == 1, :]

    # Make vector plot if selected variable is wind variable
    if variable == "U":
        grid_U, grid_V = transform_to_2D_grid(df_foehn, variable, variable_lvl, lats_labels, lons_labels)
        create_vectorfield(grid_U, grid_V, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels, df_importances, location)
    else:  # Else make contour plot
        grid = transform_to_2D_grid(df_foehn, variable, variable_lvl, lats_labels, lons_labels)
        create_contour(grid, variable, variable_lvl, unit, model, vmin, vmax, lats_labels, lons_labels, df_importances, location)
        
    
if __name__ == "__main__":
    pass