# run_simulation_comparison.py

import os
from pathlib import Path
from simulate_case import simulate_case

# Import the specific plotting functions that are useful for comparing two scenarios.
# We will create a full suite of comparative plots.
from engines.visualize import (
    plot_stock_flow,
    plot_hydrograph_ensemble,
    plot_response_spyder,
    plot_hillslope_ash_decay
)

# --- 1. Set Up Script Environment ---
# Ensure the script runs from its own directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory set to: {os.getcwd()}")


# --- 2. Define Simulation Parameters ---
# This is the base configuration that will be used for BOTH scenarios.
# The `seed` is crucial for ensuring identical rainfall in both runs.
# The eruption day is set to the middle of the year for a good baseline comparison.
base_cfg = dict(
    hydro_model       = "scscn",          # Use the hourly SCS-CN model
    num_days          = 365,              # Length of the simulation
    seed              = 42,               # Keep seed the same for identical rainfall
    
    # Basin & Hydrology Parameters
    dem_path          = "./data/RangitaikiTarawera_25m.tif",
    min_stream_order  = 8,
    clean_net         = True,
    curve_number      = 70,
    channel_velocity_ms = 0.5,
    initial_abstraction_ratio = 0.05,
    wash_efficiency   = 0.05,             # A slightly higher efficiency for clearer erosion
    transport_coefficient = 5e-4,
    max_cn_increase   = 25.0,
    k_ash             = 0.5,
    
    # Precipitation Parameters
    mean_annual_precip= 2500,
    rain_prob         = 0.5,
    season_strength   = 0.20,
    precip_sigma      = 5.0,

    # Volcano & Ash Parameters
    volcano_x         = 1852398,
    volcano_y         = 5703897,
    rho_ash           = 1000.0,
    gamma0            = 0.12,
    eta0              = 1.2,
    wind_mu           = 12,
    wind_sigma        = 3,
)


# --- 3. Create the Two Scenarios: With Ash vs. No Ash ---

# SCENARIO 1: With Ash
# We create a full copy of the base config and set the eruption parameters.
cfg_ash = base_cfg.copy()
cfg_ash['tag'] = "SCENARIO_WITH_ASH"
cfg_ash['M0'] = 4.0  # Magnitude 4 eruption to generate ash
cfg_ash['eruption_days'] = [180] # Eruption occurs on day 180

# SCENARIO 2: No Ash (Control Run)
# We create another copy and disable the eruption.
cfg_no_ash = base_cfg.copy()
cfg_no_ash['tag'] = "SCENARIO_NO_ASH"
cfg_no_ash['M0'] = -1.0 # A negative magnitude ensures no ash is generated
# No eruption_days needed as there is no eruption.

# --- 4. Run Both Simulations ---
# This will run the two scenarios back-to-back.

print("\n--- RUNNING SCENARIO 1: WITH ASH ---")
results_ash, precip_df = simulate_case(cfg_ash) # We only need precip_df once

print("\n--- RUNNING SCENARIO 2: NO ASH (CONTROL) ---")
results_no_ash, _ = simulate_case(cfg_no_ash)


# --- 5. Generate Comparative Plots ---
# All plots will be saved to a single, clearly-named directory.

output_dir = Path("outputs") / "A_MAIN_COMPARISON_RESULTS"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n--- GENERATING COMPARATIVE PLOTS in '{output_dir}' ---")

# Plot 1: Spyder chart to compare overall sensitivity
plot_response_spyder(
    results=results_ash, # The function uses the eruption day from this result
    precip_df=precip_df,
    outfile=output_dir / "1_comparison_spyder_chart.png"
)

# Plot 2: Ensemble hydrograph to compare average flood shapes
plot_hydrograph_ensemble(
    results=results_ash,
    precip_df=precip_df,
    outfile=output_dir / "2_comparison_hydrograph_ensemble.png"
)

# Plot 3: Stock and Flow for the "With Ash" scenario
plot_stock_flow(
    precip_df=precip_df, 
    results=results_ash, 
    outfile=output_dir / "3a_stock_flow_WITH_ASH.png", 
    tag=cfg_ash['tag']
)

# Plot 4: Stock and Flow for the "No Ash" scenario for reference
plot_stock_flow(
    precip_df=precip_df, 
    results=results_no_ash, 
    outfile=output_dir / "3b_stock_flow_NO_ASH.png", 
    tag=cfg_no_ash['tag']
)

# Plot 5: Hillslope ash decay for the "With Ash" scenario
plot_hillslope_ash_decay(
    results=results_ash,
    precip_df=precip_df,
    outfile=output_dir / "4_hillslope_ash_decay.png"
)

print("\n--- COMPARISON RUN FINISHED ---")
