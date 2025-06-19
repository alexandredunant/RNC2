# ============================================================
#  run_simulations_nz_pastoral.py  –  SCS-CN for NZ Pastoral Land
# ============================================================

import os
from itertools import product
from simulate_case import simulate_case
from tqdm import tqdm

# Ensure working directory is script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
print(f"Current working directory set to: {os.getcwd()}")

# ------------------------------------------------------------------
# ❶ Build parameter dictionaries for NZ Pastoral Land
# ------------------------------------------------------------------

# Typical NZ North Island rainfall
mean_rain_values = [2500]  # mm/year - typical range for North Island

# SCS-CN values for pastoral land in New Zealand
# Based on hydrologic soil groups and land use:
# - Good pasture condition: CN 61-74 (HSG B-C)
# - Fair pasture condition: CN 69-79 (HSG B-C)
# - Poor pasture condition: CN 79-86 (HSG B-C)
# For relatively flat pastoral land with moderate drainage (HSG B-C):
curve_number_values = [70]  # Good to fair pasture condition

# Channel velocity for lowland streams
# NZ pastoral streams typically have lower velocities
channel_velocity_values = [0.5]  # m/s - typical for lowland pastoral streams

# Other parameters
season_strength_values = [0.20]  # NZ has moderate seasonality
M0_values = [1.0]  # eruption magnitude

# Initial abstraction ratio
# Standard is 0.05, but for pastoral land with some compaction, wet, temperate climate 
ia_ratio_values = [0.05]

# Wash scaling value of ash with surface flow
wash_efficiency_values = [0.2]

# Add a new list for transport coefficient values to test
transport_coefficient_values = [5e-4]

# CN multiplier values in case of ash
ash_cn_multiplier_values = [10.0]

parameter_sets = []
for cn, vel, ia_ratio, ss, rain, M0, weff, tcoeff, ash_mult in product(
    curve_number_values, 
    channel_velocity_values,
    ia_ratio_values,
    season_strength_values, 
    mean_rain_values, 
    M0_values,
    wash_efficiency_values,
    transport_coefficient_values,
    ash_cn_multiplier_values
):
    parameter_sets.append(dict(
        tag               = f"CN{cn}_V{vel}_IA{ia_ratio}_S{ss}_R{rain}_M{M0}_WE{weff}_TC{tcoeff}_AM{ash_mult}",
        # ------------- hydrology model selection -------------
        hydro_model       = "scscn",  # Use SCS-CN model
        curve_number      = cn,       # SCS Curve Number for pastoral land
        channel_velocity_ms = vel,    # Lowland stream velocity
        initial_abstraction_ratio = ia_ratio,  # Ia/S ratio
        # ------------- precipitation -------------------------
        num_days          = 365, 
        mean_annual_precip= rain,
        rain_prob         = 0.5,      
        season_strength   = ss,
        precip_sigma      = 5.0,      # Moderate variability
        # ------------- eruption / ash ------------------------
        M0                = M0,
        wind_mu           = 12,
        wind_sigma        = 3,
        gamma0            = 0.12,
        eta0              = 1.2,
        rho_ash           = 1000.0,
        x_volcano         = 1852398,
        y_volcano         = 5703897,
        wash_efficiency   = weff,
        transport_coefficient = tcoeff,
        ash_cn_multiplier = ash_mult,
        # ------------- network / hydro -----------------------
        dem_path          = "./data/RangitaikiTarawera_25m.tif",
        min_stream_order  = 8,
        # These are for sediment transport (less important for hydrology)
        k                 = 0.03,
        n                 = 1.5,
        baseflow_coeff    = 5e-3,     # Higher baseflow for NZ conditions
        seed              = 42,
        clean_net         = True,
        evapotranspiration= 2.5       # Lower ET for pastoral grass (mm/day)
    ))

# ------------------------------------------------------------------
# ❷ Summary and execution
# ------------------------------------------------------------------
print(f"\n NUMBER OF PARAMETER SETS: {len(parameter_sets)}")
print("\nParameter ranges for NZ pastoral land:")
print(f" - Curve Numbers: {sorted(set(p['curve_number'] for p in parameter_sets))}")
print(f" - Channel velocities: {sorted(set(p['channel_velocity_ms'] for p in parameter_sets))} m/s")
print(f" - Mean annual rainfall: {sorted(set(p['mean_annual_precip'] for p in parameter_sets))} mm")
print(f" - Initial abstraction ratios: {sorted(set(p.get('initial_abstraction_ratio', 0.2) for p in parameter_sets))}")

print("\nSCS-CN Guidelines for context:")
print(" - CN 65: Good pasture, >75% ground cover")
print(" - CN 72: Fair pasture, 50-75% ground cover")  
print(" - CN 78: Poor pasture, <50% ground cover")
print(" - Typical NZ pastoral land: CN 65-75")

print("\nStarting simulations...\n")
print("=" * 80)

for cfg in tqdm(parameter_sets, desc="Running simulations"):
    print(f"\nRunning: {cfg['tag']}")
    simulate_case(cfg)
    
print("\n --- ALL SIMULATIONS FINISHED ---")