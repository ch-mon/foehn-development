import dypy.netcdf as dn
import numpy as np
## Constants for ERAI
# W-E direction
LON_MIN_ERA= 0
LON_MAX_ERA = 15

# S-N direction
LAT_MIN_ERA= 42
LAT_MAX_ERA = 50

path = "/net/litho/atmosdyn/INTEXseas/cesm/cesm112_LENS/b.e112.B20TRLENS.f09_g16.ethz.001/archive/atm/hist/b.e112.B20TRLENS.f09_g16.ethz.001.cam.h2.1990-01-01-21600.nc"

lons, lats = dn.read_var(path, ["lon", "lat"])

xindex = np.where((lons >= LON_MIN_ERA) & (lons <= LON_MAX_ERA))[0]
yindex = np.where((lats >= LAT_MIN_ERA) & (lats <= LAT_MAX_ERA))[0]

xmin, xmax = xindex.min(), xindex.max()
ymin, ymax = yindex.min(), yindex.max()

LATS_CESM = lats[yindex]
LONS_CESM = lons[xindex]

INDEX_CESM = np.s_[:, :, ymin:(ymax+1), xmin:(xmax+1)]

LATS_LABELS_CESM = [str(int(100*lat)) for lat in LATS_CESM]
LONS_LABELS_CESM = [str(int(100*lon)) for lon in LONS_CESM]

