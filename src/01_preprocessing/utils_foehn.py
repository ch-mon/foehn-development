import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import seaborn as sns
 
# Month abbreviation
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Get base directory which is two levels up
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))

def save_figure(name):
    filepath = os.path.join(BASE_DIR, "figures", f"{name}.pdf")
    plt.savefig(filepath, bbox_inches='tight', dpi=200)
    print(f'Saved figure at: {filepath}')


def generate_aggregated_timeseries(df):
    """
    Down-sample foehn time series from ten minutes to six hour intervals.
    @param df: Dataframe with original values (ten minute resolution)
    @type df: pd.DataFrame
    @return: Dataframe with six hour resolution
    @rtype: pd.DataFrame
    """

    # Define rolling window of length 6
    foehn_rolling_window = df["Foehn"].rolling(window=6, min_periods=4).sum().shift(-3)  # Allow max 2 missing values (min_periods=4)
    foehn_new_representation = (foehn_rolling_window >= 4).astype(int)  # If at least 4 samples show foehn prevalance, define as foehn, otherwise not (refer to Gutermann et. al., 2013)
    foehn_new_representation.loc[foehn_rolling_window.isnull()] = np.NaN  # Carry over missing values

    # Overwrite foehn column
    df["Foehn"] = foehn_new_representation

    # Keep only timestamps at full hour and where hour equals 0, 6, 12, or 18
    date_mask = (df["date"].dt.minute == 0) & \
                ((df["date"].dt.hour == 0) |
                 (df["date"].dt.hour == 6) |
                 (df["date"].dt.hour == 12) |
                 (df["date"].dt.hour == 18))
    df = df.loc[date_mask]

    # Sanity check
    print(df["Foehn"].value_counts(normalize=False, dropna=True))
    display(df)

    return df


def calculate_foehn_length(df):
    """
    Calculate mean duration and number of foehn events.
    @param df: Dataframe with 10 minute resolution
    @type df: pd.DataFrame
    """
    counter = 0
    foehn_durations = []  # List of all foehn events
    for i in range(1, len(df.index) - 1):
        print(i, end="\r")
        if (df.loc[i, "Foehn"] == 1):
            counter += 1
        elif (df.loc[i-1, "Foehn"] == 1) and (df.loc[i, "Foehn"] == 0) and (df.loc[i+1, "Foehn"] == 1):  # Allow breaks of 10 mins
            counter += 1
        else:
            if counter:  # Append if foehn ends
                foehn_durations.append(counter*10)
                counter = 0

    print("Foehn mean duration: ", np.array(foehn_durations).mean())
    print("Number of foehn events: ", len(foehn_durations))


def plot_monthly_foehn_distribution(df, location):
    """
    Calculate mean monthly foehn frequency from ten minute resolution data
    @param df: Dataframe with 10 minute resolution
    @type df: pd.DataFrame
    """
    # Create month and year column
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year

    # Group dataframe by dataset, ensemble, year, month, and prediction and calculate mean
    df_foehn_frequency = df[["year", "month", "Foehn"]].groupby(["year", "month"], axis=0, as_index=False).mean()

    # Monthly foehn frequency distribution plot
    plt.figure()
    fig = sns.boxplot(x="month", y="Foehn", data=df_foehn_frequency, color="tab:blue")
    fig.set_xticklabels(MONTH_NAMES);
    plt.xlabel("")
    plt.ylabel("Monthly mean foehn frequency")
    save_figure(f"foehn_climatology_{location}")
