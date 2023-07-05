#!/usr/bin/env python3

import numpy as np

from mods_trigo import square, distance, bearing

alpha = 0.93
beta = 0.19


def volcanic_activity_probability(a, b, r):
    """
	Calculate probability of volcanic eruption
	"""
    f = 1 - np.exp(-(a * r) ** b)
    return f


def time_to_next_volcanic_activity(a, b, s):
    """
	if time of previous explosion is s
	then following explosion will be s + R
	"""
    import math

    r = (1 / a) - math.log(np.random.uniform(0, 1) ** (1 / b))

    return r


def geometry_volcano_catchment(x_volcano,
                               y_volcano,
                               catchment_geometries):
    """
	calculate distance and bearings 
	volcanic 'epicenter' to catchment centroids
	"""
    catchment_centroids = catchment_geometries.centroid
    x_catchment = catchment_centroids.x
    y_catchment = catchment_centroids.y

    Vx = (x_catchment - x_volcano)
    Vy = (y_catchment - y_volcano)

    distances = distance(Vx, Vy)
    bearings = bearing(Vx, Vy)

    return distances, bearings


def ash_volumes_dipersed(col_height_alpha, col_height_beta):
    column_height_km = np.random.weibull(col_height_alpha) * col_height_beta
    V = 2.6 * np.exp(0.065 * column_height_km - 1.69)

    return V, column_height_km


def ash_thickness(ash_thickness_at1km,
                  diff_speed,
                  distance_volc,
                  dispersal_angle,
                  wind_direction,
                  column_height_km):
    """
	Calculate the ash thicknesses at points 
	away from volcanic center
	column height is generated randmly 
	from Weibull distribution with parameters a = 0.93, b = 0.19
	"""
    decay = 2.5 - 0.05 * column_height_km
    ash_thickness_mean = ash_thickness_at1km * np.exp(
        -diff_speed * distance * [1 - np.cos(dispersal_angle - wind_direction)]) * distance_volc ** -decay

    Z = np.random.normal()
    ash_thickness = np.exp(0.22 * Z + ash_thickness_mean)

    return ash_thickness
