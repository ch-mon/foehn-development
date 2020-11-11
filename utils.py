def calc_pot_temp(T, p):
    p_0 = 1000.0
    c_p = 1005.0
    R = 287.0

    pot_temp = (T)*(p_0/p)**(R/c_p)
    return pot_temp