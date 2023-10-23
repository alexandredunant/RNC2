
def simulate_river_flow(river_network, water_flow, num_iterations, loss_factor):
    """
    Simulate the flow of water in a river network using a graphical method.

    Args:
    - river_network: a dictionary representing the river network where the keys are the node names and the values are lists of edges, where each edge is a tuple of the form (target_node, flow_rate).
    - water_flow: a dictionary representing the initial water flow at each node.
    - num_iterations: the number of iterations to simulate.
    - water_loss_factor: a factor representing the loss of water due to evaporation, absorption, etc.

    Returns:
    - a dictionary representing the water flow at each node after the simulation.
    """

    for iteration in range(num_iterations):
        # Calculate the water flow at each node in the next iteration
        next_water_flow = {}

        for node in river_network:
            # Calculate the total inflow of water to this node from river stream and catchment flow
            target_node = river_network[node]['target_river']
            edge_flow = river_network[node]['edge_flow']

            if node in water_flow :
                if iteration == 0: # Add the initial discharge of the river stream
                    inflow = edge_flow + water_flow[node]
                else:  # Then add only the constant inflow from the catchments (otherwise the baseline flow will
                    # stack up)
                    inflow = water_flow[node] * (loss_factor) # increase water loss as time goes by

            next_water_flow[target_node] = inflow
            next_water_flow[1] = 0

        # Update the water flow at each node for the next iteration
        # At this point, the catchment flow has provided inflows and only the river flow remains
        water_flow = next_water_flow

    return water_flow





def visualize_river_flow(river_network, water_flow, node_positions):
    """
    Visualize the flow of water in a river network using a graphical method.

    Args:
    - river_network: a dictionary representing the river network where the keys are the node names and the values are lists of edges, where each edge is a tuple of the form (target_node, flow_rate).
    - water_flow: a dictionary representing the water flow at each node.
    - node_positions: a pandas DataFrame representing the x, y coordinates of each node, where the index is the node name and the columns are 'x' and 'y'.
    """

    for node, pos in node_positions.iterrows():
        plt.text(pos['x'], pos['y'], f"{node}\n{water_flow[node]:.2f} m3/s", ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))
        plt.plot(pos['x'], pos['y'], 'o')
    for node, edges in river_network.items():
        for edge in edges:
            source_node, edge_flow = edge
            source_pos = node_positions.loc[source_node]
            target_pos = node_positions.loc[node]
            plt.arrow(source_pos['x'], source_pos['y'], target_pos['x']-source_pos['x'], target_pos['y']-source_pos['y'], length_includes_head=True, head_width=0.1, head_length=0.2, alpha=0.5, color='blue', width=edge_flow*0.02)
    plt.xlim(node_positions['x'].min() - 1, node_positions['x'].max() + 1)
    plt.ylim(node_positions['y'].min() - 1, node_positions['y'].max() + 1)
    plt.axis('off')
    plt.show()
