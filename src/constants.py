import dypy.netcdf as dn
import numpy as np
import os

# Get base directory which is two levels up (from where this is called)
BASE_DIR = os.path.dirname(os.path.dirname(os.getcwd()))

# Month abbreviation
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Define box within to consider grid points in ERAI and CESM (in degrees)
LON_MIN = 0
LON_MAX = 15
LAT_MIN = 42
LAT_MAX = 50

# Create ERAI coordinates
LATS_ERA = range(LAT_MIN, LAT_MAX+1)
LONS_ERA = range(LON_MIN, LON_MAX+1)

# Load lat and lon coordinates from a exemplary CESM file
path = "/net/litho/atmosdyn/INTEXseas/cesm/cesm112_LENS/b.e112.B20TRLENS.f09_g16.ethz.001/archive/atm/hist/b.e112.B20TRLENS.f09_g16.ethz.001.cam.h2.1990-01-01-21600.nc"
lons, lats = dn.read_var(path, ["lon", "lat"])

# Get index of coordinates in bounding box
xindex = np.where((lons >= LON_MIN) & (lons <= LON_MAX))[0]
yindex = np.where((lats >= LAT_MIN) & (lats <= LAT_MAX))[0]
xmin, xmax = xindex.min(), xindex.max()
ymin, ymax = yindex.min(), yindex.max()

# Define a slice which masks all values in the bounding box
LATS_CESM = lats[yindex]
LONS_CESM = lons[xindex]

INDEX_CESM = np.s_[:, :, ymin:(ymax+1), xmin:(xmax+1)]

LATS_CESM_STRING = [str(int(100*lat)) for lat in LATS_CESM]
LONS_CESM_STRING = [str(int(100*lon)) for lon in LONS_CESM]

print("--- Avaliable variables ---")
print("Base directory: BASE_DIR")
print("Month names: MONTH_NAMES")
print("Bounding box coordinates: LON_MIN, LON_MAX, LAT_MIN, LAT_MAX")
print("ERAI coordinates: LONS_ERA, LATS_ERA")
print("CESM coordinates: LONS_CESM, LATS_CESM")
print("CESM slice: INDEX_CESM")
print("CESM coordinates for plotting (string): LONS_CESM_STRING, LATS_CESM_STRING")
