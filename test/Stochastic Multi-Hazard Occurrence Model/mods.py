import numpy as np

# probability of rain Sep-Nov, Dec-Feb, Mar-May, Jun-Aug
rain_str = ['Sep-Nov', 'Dec-Feb', 'Mar-May', 'Jun-Aug']
rain_proba = [0.43, 0.4, 0.35, 0.39]
# intensity of rainfall
rain_m = [1.62, 1.56, 1.73, 1.8]
rain_s = [1.01, 1.06, 1.09, 1.07]


def generate_hourly_rainfall_amount(probability_rain, mu, sigma):
    u = np.random.uniform(0, 1)
    
    if probability_rain < u:
        z = np.random.normal()
        amount_of_daily_rainfall = u * np.exp(mu + sigma * z)
        amount_of_hourly_rainfall = amount_of_daily_rainfall / 24
    else:
        amount_of_hourly_rainfall = 0
    return amount_of_hourly_rainfall
 
 
def calculate_effective_rainfall(ssat, rsa, rt, f1, fsa):
    # primary run off F1 state
    if ssat < rsa:
        reff = f1 * rt
    # secondary saturated runoff Fsa state
    else:
        reff = f1 * rt + (fsa - f1) * (ssat - rsa) * rt

    return reff


def calculate_saturation_in_catchment(ssat, reff, discharge):
    if reff != 0:
        sat = ssat + reff - discharge
        if sat < 0:
            sat = 0
    else:
        sat = 0
    return sat


def calculate_catchment_total_discharge(reff, k_factor, catchment_area, baseflow, time_period=1, p=1):
    unitary_discharge = (reff / k_factor) ** (1 / p)
    total_discharge = (unitary_discharge * catchment_area) / time_period + baseflow
    return total_discharge
