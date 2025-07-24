# run_simulation_comparison.py

import os
from pathlib import Path
from simulate_case import simulate_case

# Import the specific plotting functions that are useful for comparing two scenarios.
from engines.visualize import (
    plot_stock_flow,
    plot_hydrograph_ensemble,
    plot_response_spyder,
    plot_hillslope_ash_decay
)

# --- 1. Set Up Script Environment ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory set to: {os.getcwd()}")


# --- 2. Define Simulation Parameters ---
# This is the base configuration that will be used for BOTH scenarios.
base_cfg = dict(
    hydro_model       = "scscn",
    num_days          = 365,
    seed              = 42,

    # Basin & Hydrology Parameters
    dem_path          = "./data/RangitaikiTarawera_25m.tif",
    min_stream_order  = 8,
    clean_net         = True,
    curve_number      = 70,
    channel_velocity_ms = 0.5,
    initial_abstraction_ratio = 0.05,
    wash_efficiency   = 0.05,
    transport_coefficient = 5e-4,
    max_cn_increase   = 25.0,
    k_ash             = 0.5,

    # Precipitation Parameters
    mean_annual_precip= 2500,
    rain_prob         = 0.5,
    season_strength   = 0.20,
    precip_sigma      = 5.0,

    # Volcano & Ash Parameters
    volcano_x         = 1900802,
    volcano_y         = 5703897,
    rho_ash           = 1000.0,
    gamma0            = 0.0016,
    alpha0            = 794.0,
)


# --- 3. Create the Two Scenarios: With Ash vs. No Ash ---

# SCENARIO 1: With Ash
# To use a predefined eruption, set the flag and provide a simple list of days.
# The magnitude is now handled stochastically inside the ashfall engine.
cfg_ash = base_cfg.copy()
cfg_ash['tag'] = "SCENARIO_WITH_ASH"
cfg_ash['use_predefined_eruptions'] = True
cfg_ash['predefined_eruptions_list'] = [180]  # Eruption occurs on day 180

# SCENARIO 2: No Ash (Control Run)
# To disable eruptions, provide an empty list.
cfg_no_ash = base_cfg.copy()
cfg_no_ash['tag'] = "SCENARIO_NO_ASH"
cfg_no_ash['use_predefined_eruptions'] = True
cfg_no_ash['predefined_eruptions_list'] = [] # An empty list ensures no eruptions

# --- 4. Run Both Simulations ---
print("\n--- RUNNING SCENARIO 1: WITH ASH ---")
results_ash, precip_df = simulate_case(cfg_ash)

print("\n--- RUNNING SCENARIO 2: NO ASH (CONTROL) ---")
results_no_ash, _ = simulate_case(cfg_no_ash)


# --- 5. Generate Comparative Plots ---
output_dir = Path("outputs") / "A_MAIN_COMPARISON_RESULTS"
output_dir.mkdir(parents=True, exist_ok=True)
print(f"\n--- GENERATING COMPARATIVE PLOTS in '{output_dir}' ---")

if results_ash.get('eruption_days') and results_ash['eruption_days']:
    print(f"Eruption successfully simulated on day(s): {results_ash['eruption_days']}")
    plot_response_spyder(
        results=results_ash,
        precip_df=precip_df,
        outfile=output_dir / "1_comparison_spyder_chart.png"
    )
    plot_hydrograph_ensemble(
        results=results_ash,
        precip_df=precip_df,
        outfile=output_dir / "2_comparison_hydrograph_ensemble.png"
    )
    plot_hillslope_ash_decay(
        results=results_ash,
        precip_df=precip_df,
        outfile=output_dir / "4_hillslope_ash_decay.png"
    )
else:
    print("Skipping eruption-dependent plots as no eruption data was found.")

plot_stock_flow(
    precip_df=precip_df,
    results=results_ash,
    outfile=output_dir / "3a_stock_flow_WITH_ASH.png",
    tag=cfg_ash['tag']
)

plot_stock_flow(
    precip_df=precip_df,
    results=results_no_ash,
    outfile=output_dir / "3b_stock_flow_NO_ASH.png",
    tag=cfg_no_ash['tag']
)

print("\n--- COMPARISON RUN FINISHED ---")