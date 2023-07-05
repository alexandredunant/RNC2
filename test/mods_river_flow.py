#!/usr/bin/env python3
import matplotlib.pyplot as plt
import os
import imageio


def advance_river_flow(River_nodes):
	'''
	advance the river flow along the river nodes
	by a certain amount of time ticks
	'''
	
	import numpy as np
	import geopandas as gpd
	
	RIVER_id = River_nodes["rid"].values
	RIVER_target = River_nodes["TargetID"].values
	RIVER_flows = River_nodes["flows"].values
	F = np.zeros(River_nodes.shape[0], np.float64)
	
	# update new flow value at river nodes
	# based on up stream flow
	for i,_ in enumerate(RIVER_id):
		
		newflow = RIVER_flows[RIVER_target == RIVER_id[i]]
		
		if newflow:
			F[i] = newflow
			
	return F


def catchment_runout_to_stream(Cacthment_nodes, River_nodes):
    '''
    calculate runout from cacthment 
    to the closest river node
    '''
    areas = Cacthment_nodes.area
    

# # test flow
# R["flows"] = 10
# R.iloc[0,5] = 1000

# for i in range(100):
#     plt.scatter(R.x, R.y,s=R["flows"]/10)
#     plt.savefig(f"../test/test_{i}.png")
#     plt.close()
#     R["flows"]=advance_river_flow(R)
    

# png_dir = '../test'
# images = []
# for file_name in sorted(os.listdir(png_dir)):
#     if file_name.endswith('.png'):
#         file_path = os.path.join(png_dir, file_name)
#         images.append(imageio.imread(file_path))

# # Create gif
# # https://ezgif.com/gif-to-mp4/ezgif-4-37cede0c1a.gif
# for _ in range(30):
#     images.append(imageio.imread(file_path))

# imageio.mimsave('../test/river_movie.gif', images)