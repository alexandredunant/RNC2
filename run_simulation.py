# run_simulation.py

import os
from pathlib import Path
from simulate_case import simulate_case
from tqdm import tqdm

# Import all the plotting functions
from engines.visualize import (
    plot_precip_discharge_relationship,
    plot_stock_flow,
    network_animation,
    plot_event_hydrographs,
    plot_runoff_coefficient,
    plot_hillslope_ash_decay,
    plot_hydrograph_ensemble,
)

# --- 1. Set Up Script Environment ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory set to: {os.getcwd()}")

# --- 2. Build the parameter dictionary for a single simulation ---
cfg = dict(
    tag="RUN_WITH_ASHFALL",

    # --- DEM Configuration ---
    dem_path="./data/RangitaikiTarawera_25m.tif",

    # --- Model and Simulation Length ---
    hydro_model="scscn",
    num_days=1825,  # 5 years
    seed=42,

    # --- Ashfall & Volcano Parameters ---
    use_predefined_eruptions=False,
    predefined_eruptions_list=[],
    # Parameters to control eruption frequency
    weibull_shape=1.0,
    weibull_scale=250.0, 

    # Ash Dispersal Baseline Coefficients
    gamma0=0.0016,
    alpha0=794.0,

    # Volcano Location
    volcano_x=1900802,
    volcano_y=5703897,

    # Physical property of ash
    rho_ash=1000.0,

    # --- Basin & Hydrology Parameters ---
    min_stream_order=8,
    clean_net=True,
    curve_number=70,
    channel_velocity_ms=0.5,
    initial_abstraction_ratio=0.05,
    wash_efficiency=0.05,
    transport_coefficient=5e-4,
    max_cn_increase=25.0,
    k_ash=0.5,

    # --- Precipitation Parameters ---
    mean_annual_precip=2500,
    rain_prob=0.3,
    season_strength=0.20,
    precip_sigma=5.0,
)

# --- 3. Run the simulation ---
print(f"\n--- RUNNING SIMULATION: {cfg['tag']} ---")
results, precip_df = simulate_case(cfg)

# --- 4. Generate all plots ---
print("\n--- GENERATING PLOTS ---")
output_dir = Path("outputs") / cfg['tag']
output_dir.mkdir(parents=True, exist_ok=True)

# Call all the plotting functions
plot_precip_discharge_relationship(precip_df, results, output_dir / "precip_discharge_relationship.png", tag=cfg['tag'])
plot_stock_flow(precip_df, results, output_dir / "stock_flow.png", tag=cfg['tag'])
if results.get('eruption_days'):
    if results['eruption_days']:
        plot_event_hydrographs(results, precip_df, output_dir / "event_hydrograph_comparison.png", ash_event_day=results['eruption_days'][0])
    plot_hydrograph_ensemble(results, precip_df, output_dir / "hydrograph_ensemble.png")

plot_runoff_coefficient(results, precip_df, output_dir / "runoff_coefficient.png")
plot_hillslope_ash_decay(results, precip_df, output_dir / "hillslope_ash_decay.png")
network_animation(results, output_dir / "network_animation.mp4")

print("\n--- SIMULATION FINISHED ---")
