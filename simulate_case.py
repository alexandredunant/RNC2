# simulate_case.py

from pathlib import Path
import json
import numpy as np

# Core engine functions and DEM utilities
from engines import (
    generate_precip,
    generate_ashfall,
    load_dem_from_file,
    build_from_dem,
    run_nlrm_cascade,
)

def simulate_case(cfg: dict):
    """
    Runs a single simulation case based on the provided configuration.
    """
    # Create output directory
    out_dir = Path("outputs") / cfg.get("tag", "unnamed")
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- DEM handling ---
    dem_path = cfg.get("dem_path")
    dem, transform, crs = load_dem_from_file(dem_path)
    print(f"Using DEM: {dem_path}")

    # --- Build hydrology network from DEM ---
    # Pass the 'out_dir' variable to the function call to save the plot correctly
    network, basins = build_from_dem(
        dem_path=dem_path, 
        min_stream_order=cfg["min_stream_order"], 
        clean_network=cfg["clean_net"],
        output_dir=out_dir
    )
    
    # --- Generate precipitation time series ---
    precip_df = generate_precip(**cfg)
    
    # --- Generate ashfall time series ---
    # Correctly unpack the two return values from generate_ashfall
    ash_df, eruption_days_used = generate_ashfall(
        dem=dem,
        transform=transform,
        crs=crs,
        **cfg
    )

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

    print(f"DEBUG: Type of ash_df is {type(ash_df)}")

    # --- Run hydrology model ---
    # Use the explicit, correct list of arguments for this function
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
        wash_efficiency=cfg.get("wash_efficiency", 0.01),
        transport_coefficient=cfg.get("transport_coefficient", 1e-5),
        max_cn_increase=cfg.get("max_cn_increase", 25.0),
        k_ash=cfg.get("k_ash", 0.5)
    )

    # Add the used eruption days to the results for plotting
    result['eruption_days'] = eruption_days_used

    # --- Save configuration ---
    with open(out_dir / "config.json", "w") as f:
        # Make cfg serializable to save it as JSON
        serializable_cfg = {}
        for k, v in cfg.items():
            if isinstance(v, np.ndarray):
                serializable_cfg[k] = v.tolist()
            elif isinstance(v, (Path,)):
                serializable_cfg[k] = str(v)
            else:
                serializable_cfg[k] = v
        json.dump(serializable_cfg, f, indent=2)

    return result, precip_df