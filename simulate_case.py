# simulate_case.py

from pathlib import Path
import json
import numpy as np

# Core engine functions and DEM utilities
from engines import (
    generate_precip,
    generate_ashfall,
    load_dem_from_pygmt,
    save_dem_to_file,
    load_dem_from_file,
    build_from_dem,
    run_nlrm_cascade,
    timeseries,
    network_animation,
)

from engines.visualize import plot_stock_flow

def simulate_case(cfg: dict):
    """
    ... (omitting docstring for brevity) ...
    """
    # Create output directory
    out_dir = Path("outputs") / cfg.get("tag", "unnamed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- DEM handling ---
    dem_path = cfg.get("dem_path")
    if dem_path and Path(dem_path).exists():
        dem, transform, crs = load_dem_from_file(dem_path)
    else:
        # Assumes PyGMT is the alternative if dem_path is None or not found
        dem, transform, crs = load_dem_from_pygmt(cfg["region"], cfg["resolution"])
        if dem is None:
            raise RuntimeError("Failed to load DEM. Provide a valid dem_path or install PyGMT.")
        # Save the downloaded DEM for future use
        save_dem_to_file(dem, transform, crs, dem_path)
    print(f"Using DEM: {dem_path}")


    # --- Build hydrology network from DEM ---
    network, basins = build_from_dem(dem_path=dem_path, min_stream_order=cfg["min_stream_order"], clean_network=cfg["clean_net"])
    
    # --- FIX STARTS HERE ---

    # --- Generate precipitation time series ---
    # Pass only the arguments required by generate_precip
    precip_df = generate_precip(
        num_days=cfg["num_days"],
        mean_annual_precip=cfg["mean_annual_precip"],
        rain_prob=cfg["rain_prob"],
        season_strength=cfg["season_strength"],
        precip_sigma=cfg["precip_sigma"],
        seed=cfg["seed"]
    )
    
    # --- Generate ashfall time series ---
    # Pass only the arguments required by generate_ashfall
    ash_df = generate_ashfall(
        # Explicitly passed arguments
        dem=dem,
        transform=transform,
        crs=crs,
        # Arguments from the config dictionary
        num_days=cfg["num_days"],
        M0=cfg["M0"],
        wind_mu=cfg["wind_mu"],
        wind_sigma=cfg["wind_sigma"],
        gamma0=cfg["gamma0"],
        eta0=cfg["eta0"],
        seed=cfg["seed"],
        # Handle the key name mismatch: cfg['x_volcano'] -> func(volcano_x=...)
        volcano_x=cfg["x_volcano"],
        volcano_y=cfg["y_volcano"]
    )
    # --- FIX ENDS HERE ---

    # --- Select and configure hydrology model ---
    hydro_model = cfg.get("hydro_model", "nlrm")
    
    if hydro_model == "scscn":
        print("Using SCS-CN Unit Hydrograph model (hourly timesteps)")
        basins['curve_number'] = cfg.get("curve_number", 70.0)
        basins['channel_velocity_ms'] = cfg.get("channel_velocity_ms", 1.0)
        basins['initial_abstraction_ratio'] = cfg.get("initial_abstraction_ratio", 0.2)
        if 'flow_length_km' not in basins.columns:
            basins['flow_length_km'] = np.sqrt(basins['catchment_area_m2'] / 1e6)
    else:
        print("Using Non-Linear Reservoir Model (daily timesteps)")

    # --- Run hydrology model ---
    result = run_nlrm_cascade(
        edges_gdf=network,
        catch_df=basins,
        precip_df=precip_df,
        ash_df=ash_df,
        k=cfg.get("k", 0.03),
        n=cfg.get("n", 1.5),
        baseflow_coeff=cfg.get("baseflow_coeff", 5e-3),
        evapotranspiration=cfg.get("evapotranspiration", 2.5),
        rho_ash=cfg.get("rho_ash", 1000.0),
        model_type=hydro_model,
        wash_efficiency=cfg.get("wash_efficiency", 0.01), # Default to 0.01
        transport_coefficient=cfg.get("transport_coefficient", 1e-5),
        ash_cn_multiplier=cfg.get("ash_cn_multiplier", 1.0)
    )

    # --- Export outputs ---
    # timeseries(precip_df, ash_df, result, out_dir / "timeseries.png")
    network_animation(result, out_dir / "network_animation.mp4")
    plot_stock_flow(precip_df, result, out_dir / "stock_flow_relationship.png", tag=cfg.get("tag"))

    # --- Save configuration ---
    with open(out_dir / "config.json", "w") as f:
        # Make cfg serializable if it contains numpy arrays or other complex types
        serializable_cfg = {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in cfg.items()}
        json.dump(serializable_cfg, f, indent=2)

    return result