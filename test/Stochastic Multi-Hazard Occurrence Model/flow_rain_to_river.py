import pandas as pd
pd.options.mode.chained_assignment = None
from tqdm import trange

from mods import *

# load nodes
df = pd.read_csv("/Users/alexdunant/Documents/Github/rnc2_project/docs/Catchments_updt_sorted.csv")
df = df[df['River System'] == 'Rangitaki']  # keep only Rangitaiki river

# test dataframe
id=63
df = df[(df['Source_river']==id) | (df['Target_river']==id)]


df['Area (ha)'] = df['Area (m2)'] / 10000
df['saturation'] = 0  # starting saturation in each catchment
df['rsa'] = 20  # mm rainfall amount for saturation
df['river_discharge'] = np.random.randint(70, 80,size=len(df))  # mean river discharge for Rangitaiki
df['process_time'] = 0  # include internal process time

# setting period of investigation parameters
number_of_hours = 8766  # one year
season = -1

states_lists = []
hourly_rain = generate_hourly_rainfall_amount(rain_proba[0], rain_m[0], rain_s[0])

for time in trange(number_of_hours):

    # set season
    if time % np.round(8766 / 4) == 0:
        season += 1
        if season > 3:
            season = 0

    # print(f'we are in {rain_str[season]}')
    if time % 24 == 0:  # it's a new day
        # print('next day')
        hourly_rain = generate_hourly_rainfall_amount(rain_proba[season], rain_m[season], rain_s[season])

    # calculate effective rain in each catchment
    df['reff'] = df.apply(lambda row: calculate_effective_rainfall(row['saturation'],
                                                                   row['rsa'],
                                                                   hourly_rain,
                                                                   row['f1'],
                                                                   row['fsa']), axis=1)

    # artificially increase reff by 10 for example
    df['reff'] *= df['reff']

    df['total_discharge'] = df.apply(lambda row: calculate_catchment_total_discharge(row['reff'],
                                                                                     row['Proportional coefficient (K)'],
                                                                                     row['Area (ha)'],
                                                                                     row['river_discharge']), axis=1)

    df['saturation'] = df.apply(lambda row: calculate_saturation_in_catchment(row['saturation'],
                                                                              row['reff'],
                                                                              row['total_discharge']), axis=1)

    # move the flow downstream
    for i in range(len(df)):
        target = df['Target_river'].values[i]
        df['river_discharge'][i] = df[df['Source_river'] == target]['total_discharge'].sum()  # sum of all cathcment leading to river node (?)

    # database to list
    states_at_time = df[['reff', 'Catchment Name', 'total_discharge', 'saturation', 'river_discharge', 'Source_river', 'Target_river']]
    states_at_time['time'] = time
    # get recorded data over the time period
    states_lists.append(states_at_time)

# get recorded data over the total time period
results = pd.concat(states_lists)

print("Done")