from mods_weather import *
from mods_catchment import *

import pandas as pd
import numpy as np
from tqdm import trange

# probability of rain Sep-Nov, Dec-Feb, Mar-May, Jun-Aug
rain_proba = [0.43, 0.4, 0.35, 0.39]
rain_m = [1.62, 1.56, 1.73, 1.8]
rain_s = [1.01, 1.06, 1.09, 1.07]
rain_str = ['Sep-Nov', 'Dec-Feb', 'Mar-May', 'Jun-Aug']

df = pd.read_csv("../docs/Catchments_updt_sorted.csv")

saturations = np.full(0.1, len(df))
tephra_deposits = np.ones(len(df)) # ones to test algo
vegetation_param = np.ones(len(df))
df['river_flows'] = np.ones(len(df))
rsas = df['rsa'].values
f1s = df['f1'].values
fsas = df['fsa'].values
kcoefs = df['Proportional coefficient (K)'].values
areas = df['Area (m2)'].values
lengths = df['Length'].values
slopes = df['Slope (Gradient)'].values



number_of_hours = 8766  # one year
season = -1

states_lists = []

for time in trange(number_of_hours):

    # weather
    if time % np.round(8766/4) == 0:
        season += 1
        if season > 3:
            season = 0
            
    # print(f'we are in {rain_str[season]}')

    if time % 24 == 0: # it's a new day
        # print('next day')
        daily_rain = generate_daily_rainfall_amount(rain_proba[season], rain_m[season], rain_s[season])
        hourly_rain = daily_rain / 24 # we assume same intensity every hour for 24h ...
    else:
        hourly_rain = daily_rain / 24

    # effective rain for each catchment
    effective_rains = [effective_rainfall(saturations[i],
                                          rsas[i],
                                          hourly_rain,
                                          f1s[i],
                                          fsas[i]) for i in range(len(df))]

    # calculate catchment discharges to river
    catchment_discharges = [catchment_discharge(effective_rains[i],
                                                kcoefs[i],
                                                areas[i],
                                                df['river_flows'][i]) for i in range(len(df))]

    # update saturation "state" for each catchment
    saturations = [catchment_saturation(saturations[i],
                                       effective_rains[i],
                                       catchment_discharges[i]) for i in range(len(df))]

    # calculate tephra runoff to river and take minimum value of tephra and possible run off
    tephra_runoffs = [tephra_runoff(catchment_discharges[i],
                                    areas[i],
                                    lengths[i],
                                    slopes[i],
                                    vegetation_param[i]) for i in range(len(df))]

    tephra_to_stream = list(map(min, zip(tephra_runoffs, tephra_deposits)))
    
    # move the flow downstream
    df['river_flows'] = catchment_discharges
    # (might need a time / delay equation from catchment to river)
    new_flows = np.zeros(len(df))
    
    for i in range(len(df)):
        
        if df['River System'][i] == 'Tarawera':
            
            target = df['Target_river'][i]
            new_flows[i] = df[df['Source_river']==target]['river_flows'].max()
        
        elif df['River System'][i] == 'Rangitaki':
            
            target = df['Target_river'][i]
            new_flows[i] = df[df['Source_river']==target]['river_flows'].max()
    
    # database to list
    states_at_time = pd.DataFrame(
        {'time': time,
         'effective_rain': effective_rains,
         'catchment_discharges': catchment_discharges,
         'catchment_saturations': saturations,
         'tephra_runoff': tephra_runoffs,
         'River System' : df['River System'],
         'Source River' : df['Source_river'],
         'Target River' : df['Target_river'],
         'river_flows' : new_flows,
         })

    # get recorded data over the time period
    states_lists.append(states_at_time)
    
# get recorded data over the total time period
results = pd.concat(states_lists)