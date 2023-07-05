#!/usr/bin/env python3

# probability of rain Sep-Nov, Dec-Feb, Mar-May, Jun-Aug
rain_proba = [0.43, 0.4, 0.35, 0.39]
rain_m = [1.62, 1.56, 1.73, 1.8]
rain_s = [1.01, 1.06, 1.09, 1.07]


def generate_daily_rainfall_amount(P, M, S):
	import numpy as np
	
	U = np.random.uniform(0,1)
	if U > P:
		Z = np.random.normal()
		amount_of_daily_rainfall = U * np.exp(M + S*Z)
	else:
		amount_of_daily_rainfall = 0
		
	return amount_of_daily_rainfall



def wind_direction_and_diffspeed_sampling():
	import math
	import numpy as np
	Mu = 23.3
	Mv = 0.6
	Su = 12.9
	Sv = 11.9
	p = -0.21
	
	U = Mu + np.random.normal() * Su
	V = Mv + p * (Sv/Su) * U - Mu + np.random.normal() * math.sqrt((1-p**2) * (Sv**2))
	
	wind_direction = np.degrees(math.atan(-U/V))
	wind_direction = (wind_direction + 360) % 360
	# np.degrees(np.arctan2(
	
	diff_speed = (5.1/50)* math.sqrt(U**2 + V**2)
	
	return wind_direction, diff_speed


# # test
# from matplotlib import pyplot as plt

# l_d=[]
# l_sp=[]

# for _ in range(100):
#     wind_dir, diffusion_speed = wind_direction_and_diffspeed_sampling()
#     print(wind_dir, diffusion_speed)
    
#     l_d.append(wind_dir)
#     l_sp.append(diffusion_speed)


# ax = plt.subplot(111, polar=True)
# ax.scatter(x=l_d, y=[3.6*x for x in l_sp]) # 3.6 to get km/h
# ax.set_theta_zero_location('N')
# ax.set_theta_direction(-1)