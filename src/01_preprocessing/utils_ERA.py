import dypy.intergrid as ig


def get_interpolated_variables(grid_values, index_level, SOUTH_WEST_POINT, NORTH_EAST_POINT, QUERY_POINTS):
    """
    Interpolate from ERAI grid to CESM grid
    @param grid_values: ERAI grid values for a given variable
    @type grid_values: np.array
    @param index_level: Level index of pressure level of interest
    @type index_level: int
    @param SOUTH_WEST_POINT: # Lower left corner of bounding box
    @type SOUTH_WEST_POINT: np.array
    @param NORTH_EAST_POINT: #Upper right corner of bounding box
    @type NORTH_EAST_POINT: np.array
    @param QUERY_POINTS: Query points of new CESM grid
    @type QUERY_POINTS: List of tuples
    @return: Interpolated ERAI values on CESM grid
    @rtype: np.array
    """
    grid_values_cut = grid_values[index_level]
    interfunc = ig.Intergrid(grid_values_cut, lo=SOUTH_WEST_POINT, hi=NORTH_EAST_POINT, verbose=False)
    return interfunc(QUERY_POINTS)
