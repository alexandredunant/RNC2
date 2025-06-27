# engines/rivernet.py

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import rasterio.plot
import pyflwdir
from rasterio import features
from rasterio.transform import from_origin
from affine import Affine
import networkx as nx
from pathlib import Path

def vectorize(data, nodata, transform, crs, name="value"):
    """
    Vectorize a raster into a GeoDataFrame.
    
    Args:
        data: Raster data array
        nodata: No data value
        transform: Affine transform
        crs: Coordinate reference system
        name: Name of the value column
        
    Returns:
        GeoDataFrame with vectorized features
    """
    feats_gen = features.shapes(
        data,
        mask=data != nodata,
        transform=transform,
        connectivity=8,
    )
    feats = [
        {"geometry": geom, "properties": {name: val}} for geom, val in list(feats_gen)
    ]

    # Parse to geopandas for plotting / writing to file
    gdf = gpd.GeoDataFrame.from_features(feats)
    gdf[name] = gdf[name].astype(data.dtype)
    gdf = gdf.set_crs(crs=crs)
    return gdf

def clean_river_network(river_network_gdf):
    """
    Clean the river network by removing disconnected components and outliers.
    
    Args:
        river_network_gdf: GeoDataFrame with river network
        
    Returns:
        Cleaned GeoDataFrame
    """
    import networkx as nx
    
    # Create a copy to avoid modifying the input
    network = river_network_gdf.copy()
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add all nodes (reaches)
    for rid in network['reach_id']:
        G.add_node(rid)
    
    # Add all edges (connections)
    for _, row in network.iterrows():
        rid = row['reach_id']
        ds_id = row['downstream_id']
        if pd.notna(ds_id) and ds_id != 0 and ds_id != -1:
            # Only add edges for valid downstream connections
            if ds_id in G.nodes:
                G.add_edge(rid, ds_id)
    
    # Find the largest connected component
    # For a directed graph, we use weakly connected components
    connected_components = list(nx.weakly_connected_components(G))
    
    # Sort components by size (largest first)
    connected_components.sort(key=len, reverse=True)
    
    if len(connected_components) > 1:
        largest_component = connected_components[0]
        print(f"Found {len(connected_components)} disconnected components")
        print(f"Largest component has {len(largest_component)} reaches ({len(largest_component)/len(G.nodes)*100:.1f}% of network)")
        
        # Calculate sizes of other components for reporting
        other_sizes = [len(comp) for comp in connected_components[1:]]
        if other_sizes:
            print(f"Removing {len(other_sizes)} smaller components with sizes: {other_sizes}")
        
        # Keep only reaches in the largest component
        network = network[network['reach_id'].isin(largest_component)].copy()
    else:
        print("Network is already a single connected component")
    
    # Remove every reach that is a pit or has non-positive length or slope
    invalid = (
        (network['Slope (Gradient)'] < 0)
      | (network['length'] <= 0)
      | (network['pit'])
    )
    if invalid.any():
        to_drop = network.loc[invalid, 'reach_id']
        print(f"Removing {len(to_drop)} reaches with zero/negative length or pit flag")
        network = network.loc[~invalid].copy()

    
    # Update downstream_id for any reaches that now point to non-existent reaches
    valid_reach_ids = set(network['reach_id'])
    network['downstream_id'] = network['downstream_id'].apply(
        lambda x: x if pd.notna(x) and x in valid_reach_ids else None
    )

    # Build a fresh DiGraph for cycle‐breaking
    def build_graph(df):
        G = nx.DiGraph()
        G.add_nodes_from(df['reach_id'])
        for _, row in df.iterrows():
            u = row['reach_id']
            v = row['downstream_id']
            if pd.notna(v) and v not in (0, -1):
                G.add_edge(u, v)
        return G

    network = network.reset_index(drop=True)
    G_clean = build_graph(network)

    # Iteratively find & break cycles
    while True:
        try:
            # find one cycle (list of (u,v) edges)
            cycle_edges = nx.find_cycle(G_clean, orientation='original')
        except nx.NetworkXNoCycle:
            # no more cycles
            print("Cleaned network is now acyclic")
            break

        # Pick an edge to remove; here we just take the last one in the cycle
        # (u→v), and clear u's downstream pointer
        u, v, _ = cycle_edges[-1]
        print(f"Breaking cycle by removing downstream link {u} → {v}")
        network.loc[network['reach_id'] == u, 'downstream_id'] = None

        # Rebuild the graph for the next iteration
        G_clean = build_graph(network)
    
    # Reset index to ensure it's sequential
    network = network.reset_index(drop=True)
    
    return network


# def clean_river_network(river_gdf):
#     """
#     Keep only the largest weakly‐connected component.
#     """
#     G = nx.DiGraph()
#     for rid in river_gdf["reach_id"]:
#         G.add_node(rid)
#     for _, row in river_gdf.iterrows():
#         ds = row["downstream_id"]
#         if pd.notna(ds) and ds not in (0, -1):
#             G.add_edge(row["reach_id"], int(ds))
#     comps = list(nx.weakly_connected_components(G))
#     main = max(comps, key=len)
#     return river_gdf[river_gdf["reach_id"].isin(main)]

def extract_network_from_dem(
    dem_path: str,
    min_stream_order: int = 7,
    clean_network: bool = True,
    output_dir: Path = Path(".")
):
    """
    1) Reads DEM from dem_path
    2) Extracts stream network and subbasins via pyflwdir
    3) Builds GeoDataFrames for river segments and catchments
    """
    # — load DEM
    with rasterio.open(dem_path) as src:
        elev = src.read(1).astype(float)
        transform = src.transform
        crs = src.crs

    is_geo = crs.is_geographic if hasattr(crs, "is_geographic") else False

    # — flow directions
    flw = pyflwdir.from_dem(
        data=elev,
        nodata=np.nan,
        transform=transform,
        latlon=is_geo
    )

    # — extract streams
    strahler = flw.stream_order(type="strahler")
    feats = flw.streams(min_sto=min_stream_order, 
                        strahler=strahler)
    rivers = gpd.GeoDataFrame.from_features(feats, crs=crs)
    rivers["length"] = rivers.geometry.length
    rivers = rivers.rename(
        columns={"idx": "reach_id", "idx_ds": "downstream_id", "strord": "order"}
    )

    # Extract catchments for each stream
    subbas, idxs_out = flw.subbasins_streamorder(min_sto=min_stream_order)
    catchments = vectorize(subbas.astype(np.int32), 0, transform, crs=crs, name="basin")
    catchments["catchment_area_m2"] = catchments.geometry.area
    catchments["catchment_id"] = range(1, len(catchments) + 1)
    
    # Add catchment_id to river_network using spatial join
    # Use sjoin_nearest instead of iterating through each catchment-reach pair
    catchments_for_join = catchments[['catchment_id', 'geometry']].copy()
    river_network_for_join = rivers[['reach_id', 'geometry']].copy()

    # Perform spatial join
    joined = gpd.sjoin_nearest(river_network_for_join, catchments_for_join, how='left')

    # Create mapping from reach_id to catchment_id using the joined result
    reach_to_catchment = dict(zip(joined['reach_id'], joined['catchment_id']))

    # Add catchment_id to river_network
    rivers['catchment_id'] = rivers['reach_id'].map(reach_to_catchment)
    
    # Get bounds from the transform and shape
    bounds = src.bounds
    height, width = elev.shape
    
    # Create coordinate arrays from the bounds
    x_coords = np.linspace(bounds.left, bounds.right, width)
    y_coords = np.linspace(bounds.top, bounds.bottom, height)  # Note: top to bottom for y
    
    # Verify coordinate ranges
    print(f"X range: {x_coords.min()} to {x_coords.max()}")
    print(f"Y range: {y_coords.min()} to {y_coords.max()}")
    
    # Create the xarray Dataset
    import xarray as xr
    ds = xr.Dataset(
        data_vars={
            'elevation': (('y', 'x'), elev),
        },
        coords={
            'y': y_coords,
            'x': x_coords
        },
        attrs={
            'description': 'DEM elevation with geographic coordinates',
            'crs': crs.to_string() if hasattr(crs, 'to_string') else str(crs)
        }
    )
    
    # Calculate slope
    slope = pyflwdir.dem.slope(elevtn=elev)
    ds['slope'] = (('y', 'x'), slope)
    
    # Extract Raster stats using exactextract
    from exactextract import exact_extract
    
    # For catchments
    s = ds.slope  # Use the xarray DataArray directly
    tmp = exact_extract(
        s,  # Extract numpy array from xarray
        catchments,
        ops="mean",
        output="pandas",
        include_geom="gdal",
        progress=True,
    )
    catchments["Slope (Gradient)"] = tmp["mean"]
    
    # For river segments
    tmp = exact_extract(
        s,
        rivers,
        ops="mean",
        output="pandas",
        include_geom="gdal",
        progress=True,
    )
    rivers["Slope (Gradient)"] = tmp["mean"]
    

    # — clean if requested
    if clean_network:
        rivers = clean_river_network(rivers)

    # ----------------------------
    
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

    # --- Get the geographic extent for correct axis labeling ---
    # The extent is [left, right, bottom, top] in geographic coordinates
    left, bottom, right, top = src.bounds

    # --- Plot 1: DEM with its own colorbar ---
    # Use imshow with origin='upper' and extent to ensure correct orientation
    dem_plot = ax1.imshow(elev, cmap='terrain', 
                        norm=Normalize(vmin=np.nanmin(elev), vmax=np.nanmax(elev)),
                        extent=[left, right, bottom, top], origin='upper')
    fig.colorbar(dem_plot, ax=ax1, orientation='vertical', label='Elevation (m)', shrink=0.8)
    ax1.set_title("DEM and River Network")

    # --- Plot 2: Slope with its own colorbar ---
    slope_plot = ax2.imshow(slope, cmap='cividis', 
                            norm=Normalize(vmin=0, vmax=45),
                            extent=[left, right, bottom, top], origin='upper')
    fig.colorbar(slope_plot, ax=ax2, orientation='vertical', label='Slope (degrees)', shrink=0.8)
    ax2.set_title("Slope and River Network")

    # --- Overlay river and catchment data on both plots ---
    for ax in (ax1, ax2):
        # Plot subbasin outlines
        catchments.boundary.plot(ax=ax, color='red', linewidth=0.5, alpha=0.6)
        # Plot streams with line width based on stream order
        rivers.plot(ax=ax, color='cyan', alpha=0.7, linewidth=rivers['order']/2)
        ax.set_aspect('equal')
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout(pad=1.5)
    output_path = output_dir / "DEM_river_visual_check.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved DEM visualization to: {output_path}")
    plt.close()
    # ------------------------------

    # — compute simple hydraulic geometry
    maxo = rivers["order"].max()
    rivers["manning_n"] = 0.035 + 0.015 * (maxo - rivers["order"]) / maxo
    base_w, scale = 10.0, 2.0
    rivers["width"] = base_w * (scale ** (rivers["order"] - rivers["order"].min()))

    # — subcatchment area mapping
    area_map = catchments.set_index("catchment_id")["catchment_area_m2"]
    rivers["A_sub"] = rivers["catchment_id"].map(area_map)

    return rivers, catchments

def build_from_dem(
    dem_path: str,
    min_stream_order: int = 7,
    clean_network: bool = True,
    output_dir: Path = Path(".")
):
    """
    Convenience wrapper matching the engine API.
    """
    return extract_network_from_dem(dem_path, min_stream_order, clean_network, output_dir)
