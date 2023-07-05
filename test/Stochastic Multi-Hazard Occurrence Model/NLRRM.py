import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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
