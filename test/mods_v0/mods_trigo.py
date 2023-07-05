#!/usr/bin/env python3

import numpy as np
from numba import njit

@njit
def square(x):
	return x ** 2

@njit
def distance(x, y):
	return np.sqrt(square(x) + square(y))


@njit
def bearing(x, y):
	b = np.degrees(np.arctan2(x,y))
#     degrees = (degrees + 360) % 360
	return (b + 360) % 360

