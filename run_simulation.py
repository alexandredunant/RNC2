# %%
import os
from itertools import product
from simulate_case import simulate_case
from tqdm import tqdm
from pathlib import Path

# Import the new plotting function
from engines.visualize import (
    plot_precip_discharge_relationship, 
    plot_stock_flow, network_animation, 
    plot_event_hydrographs, 
    plot_runoff_coefficient,
    plot_response_spyder,
    plot_hillslope_ash_decay,
    plot_hydrograph_ensemble
)

# Ensure working directory is script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory set to: {os.getcwd()}")

# ------------------------------------------------------------------
# Build the parameter dictionary for a single simulation
# ------------------------------------------------------------------
# This configuration will have ashfall events
cfg = dict(
    tag               = "TEST_180_eruption_new_hydo_20year",
    hydro_model       = "scscn",
    curve_number      = 70,
    channel_velocity_ms = 0.5,
    initial_abstraction_ratio = 0.05,
    num_days          = 365*20,
    mean_annual_precip= 2500,
    rain_prob         = 0.5,
    season_strength   = 0.20,
    precip_sigma      = 5.0,
    M0                = 1.0, # Eruption magnitude to ensure ashfall
    wind_mu           = 12,
    wind_sigma        = 3,
    gamma0            = 0.12,
    eta0              = 1.2,
    rho_ash           = 1000.0,
    volcano_x         = 1852398,
    volcano_y         = 5703897,
    eruption_days     = [180],
    wash_efficiency   = 0.2,
    transport_coefficient = 5e-4,
    max_cn_increase   = 25.0,
    k_ash             = 0.5,
    dem_path          = "./data/RangitaikiTarawera_25m.tif",
    min_stream_order  = 8,
    k                 = 0.03,
    n                 = 1.5,
    baseflow_coeff    = 5e-3,
    seed              = 42,
    clean_net         = True,
    evapotranspiration= 2.5,
)


# ------------------------------------------------------------------
# Run the simulation
# ------------------------------------------------------------------
print(f"\n--- RUNNING SIMULATION: {cfg['tag']} ---")
results, precip_df = simulate_case(cfg)


# ------------------------------------------------------------------
# Generate the new relationship plot
# ------------------------------------------------------------------
print("\n--- GENERATING PLOTS ---")
output_dir = Path("outputs") / cfg['tag']
output_dir.mkdir(parents=True, exist_ok=True)


plot_stock_flow(precip_df, 
                results, 
                output_dir / "stock_flow.png", 
                tag=cfg['tag']
)

plot_runoff_coefficient(
    results=results,
    precip_df=precip_df,
    outfile=output_dir / "runoff_coefficient.png"
)

plot_hillslope_ash_decay(
    results=results,
    precip_df=precip_df,
    outfile=output_dir / "hillslope_ash_decay.png"
)

network_animation(results, output_dir / "network_animation.mp4")

print("\n--- SIMULATION FINISHED ---")
# %%
