import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import trange


def generate_daily_rainfall_amount(num_days):
    # Probability of rain and intensity of rainfall for each season
    rain_str = ['Sep-Nov', 'Dec-Feb', 'Mar-May', 'Jun-Aug']
    rain_len = [30, 28, 31, 31]  # Number of days in each season
    proba_values = [0.43, 0.4, 0.35, 0.39]
    # intensity of rainfall
    mu_values = [1.62, 1.56, 1.73, 1.8]
    sigma_values = [1.01, 1.06, 1.09, 1.07]
    num_seasons = len(rain_str)

    # Determine the number of days in each season based on the ratios provided
    season_days = [int(round(n * num_days / sum(rain_len))) for n in rain_len]
    season_days[-1] = num_days - sum(season_days[:-1])

    daily_rainfall = np.zeros(num_days)
    start_day = 0

    # Generate daily rainfall amounts for each season
    for i in range(num_seasons):
        end_day = start_day + season_days[i]
        proba = proba_values[i]
        mu = mu_values[i]
        sigma = sigma_values[i]

        # Generate daily rainfall amounts for the current season
        num_season_days = end_day - start_day
        season_rainfall = np.zeros(num_season_days)

        for j in range(num_season_days):
            u = np.random.uniform(0, 1)

            if proba < u:
                z = np.random.normal()
                amount_of_daily_rainfall = u * np.exp(mu + sigma * z)
            else:
                amount_of_daily_rainfall = 0

            season_rainfall[j] = amount_of_daily_rainfall

        # Add the generated rainfall to the daily rainfall array
        daily_rainfall[start_day:end_day] = season_rainfall

        # Update the start_day for the next season
        start_day = end_day

    return daily_rainfall
    

def simulate_catchment_reservoir_flow(catchments, precipitation, dtime):
    """
    Simulates the hydrological behavior of a catchment using a non-linear reservoir model with a F1-RSA method.

    Args:
    - catchments = dataframe of the catchments
    - precipitation = a list of the precipitations for the period studied
    - time = current time 
     
        - sat = catchment['max_soil_moisture']
        - k = catchment['Proportional coefficient (K)']
        - c1R = catchment['f1']
        - c2 = catchment['fsa']
        - Rsa = catchment['rsa']
        - area = catchment['Area (m2)']
        - S = storage[i]
        - K = 0.05 hydraulic conductivity 

    Returns:
    - a dictionary representing the storage and streamflow at each catchment after the simulation.
    """
    list_S = []
    list_Sflow = []

    for i, catchment in catchments.iterrows():
        name = catchment['Catchment Name']
#       river = catchment['Target_river']
        sat = catchment['max_soil_moisture']
        k = catchment['Proportional coefficient (K)']
        c1R = catchment['f1']
        c2 = catchment['fsa']
        Rsa = catchment['rsa']
        area = catchment['Area (m2)']
        S = catchment['storage']
        Sflow = catchment['streamflow']
        K = 0.1
        
        # Calculate effective rainfall for each catchment (x the area makes the storage irrelevant ...)
        if S <= Rsa:
            Reff = c1R * precipitation #* area
        else:
            Reff = (c1R + (c2 - c1R) * ((S - Rsa) / (sat - Rsa))) * precipitation #* area
            
        # Calculate excess rainfall
        excess = max(0, Reff - (sat - S))
        
        # Calculate infiltration (Proportional coefficient - infiltration rate 'k') and update storage
        infiltration = k * S * dtime  # assuming simulation for 1 day
        S += Reff - excess - infiltration
        S = max(0, S)
        
        # Calculate streamflow using hydraulic conductivity (K)
        Sflow = K * S

        # Store results
        list_S.append(S)
        list_Sflow.append(Sflow)

    catchments['storage'] = list_S
    catchments['streamflow'] = list_Sflow

    return catchments


def catchment_flow_to_river_nodes(catchments, water_flow, baseflow):

    catchment_flow = {}
    keys = catchments['Catchment Name']
    v1 = catchments['storage'].values
    v2 = catchments['streamflow'].values
    v3 = catchments['Source_river'].values

    for i in range(len(keys)):
        catchment_flow[keys[i]] = {'storage': v1[i], 'streamflow': v2[i], 'river_node': v3[i]}

    # create dictionary with new river flows for each node
    new_water_flow = {}
    for node in water_flow:
        # get streamflow from all the catchment
        inflow = 0
        for catchment in catchment_flow:
            if catchment_flow[catchment]['river_node'] == node:
                inflow += catchment_flow[catchment]['streamflow']
        # add the current river flow + flow from the catchment
        inflow += water_flow[node]

        new_water_flow[node] = inflow + baseflow # baseflow should be 0 ...

    return new_water_flow