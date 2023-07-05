def effective_rainfall(ssat, rsa, rt, f1, fsa): # are c1 = f1 and c2 = fsa???
    """
	calculate effective rainfall for catchment with
	Rt -> Rainfall at time t
	Rsa -> Constant (level of saturation at which rainfall runs off
						at a higher rate)
	Ssat -> is the current saturation level of the ground
	f1 -> water runs off at a (lower) rate
	fsa -> water runs off when the ground is saturated (Ssat>Rsa)
	"""

    # primary run off F1 state
    if ssat < rsa:
        reff = f1 * rt
    # secondary saturated runoff Fsa state
    else:
        reff = f1 * rt + (fsa - f1) * (ssat - rsa) * rt

    return reff


def catchment_discharge(reff, k_factor, catchment_area, baseflow, time_period=1, p=1):
    """
	calculate discharge using
	Reff -> effective rainfall
	K, p -> Coefficients

	need to be multiplied by catchment area,
	divided by the time increment
	plus any baseflow
	"""
    unitary_discharge = (reff / k_factor) ** (1 / p)
    discharge = (unitary_discharge * catchment_area) / time_period + baseflow

    return discharge


def catchment_saturation(ssat, reff, discharge):
	"""
	calculate the resulting saturation
	from existing saturation
	Ssast -> existing saturation
	Reff -> effective rainfall
	Q -> discharge
	"""
	if reff != 0:
		sat = (ssat + reff - discharge)
	else:
		# saturation goes back to 0
		sat = 0
	
	return sat


def tephra_runoff(discharge, area, width, slope, vegetation_cste):
    """
	 Area -> is the catchment area
	 Width -> catchment width
	 Slope -> catchment slope
	 vegetation_cste
	 y -> tephra loss into the river 
	 """
    import numpy as np

    y = vegetation_cste * (discharge ** 1.12) * (area / width) ** 0.4 * (np.sin(slope)) ** 1.6
    return y
