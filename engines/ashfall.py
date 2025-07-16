# engines/ashfall.py

import os
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

# --- Helper functions ---
def calculate_thickness(
    distance_km: np.ndarray,
    theta_degrees: np.ndarray,
    gamma_eff: float,
    beta_u: float,
    phi_degrees: float,
    alpha_eff: float
) -> np.ndarray:
    theta_rad = np.radians(theta_degrees)
    phi_rad = np.radians(phi_degrees)
    cos_term = np.cos(theta_rad - phi_rad)
    wind_term = np.exp(-beta_u * distance_km * (1 - cos_term))
    decay_term = (distance_km + 1e-9)**(-alpha_eff)
    T_cm = gamma_eff * wind_term * decay_term
    return T_cm

def load_dem_from_pygmt(region, resolution="01s"):
    try:
        import pygmt
        grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=region)
        data, transform, crs = grid.data, from_origin(grid.coords["lon"].min(), grid.coords["lat"].max(), grid.coords["lon"][1] - grid.coords["lon"][0], grid.coords["lat"][0] - grid.coords["lat"][1]), "EPSG:4326"
        return data, transform, crs
    except ImportError:
        return None, None, None

def load_dem_from_file(dem_path):
    with rasterio.open(dem_path) as src:
        elev, transform, crs = src.read(1).astype(float), src.transform, src.crs
        if src.nodata is not None: elev[elev == src.nodata] = np.nan
        return elev, transform, crs

def save_dem_to_file(data, transform, crs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
        dst.write(data, 1)

# ──────────────── Public API ────────────────
def generate(
    *,
    num_days: int,
    seed: int,
    dem: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    volcano_x: float,
    volcano_y: float,
    use_predefined_eruptions: bool = False,
    predefined_eruptions_list: list = [],
    weibull_shape: float = 1.0,
    weibull_scale: float = 100.0,
    gamma0: float,
    alpha0: float,
    **kwargs
) -> tuple[pd.DataFrame, list]:
    rng = np.random.default_rng(seed)

    # --- 1. Determine Eruption Timing ---
    if use_predefined_eruptions:
        eruption_days = predefined_eruptions_list
        print(f"Using predefined eruption days: {eruption_days}")
    else:
        print("Generating eruption days stochastically (Weibull renewal process)...")
        eruption_days = []
        current_day = 0
        while True:
            time_to_next_days = rng.weibull(weibull_shape) * weibull_scale
            next_day = current_day + np.ceil(time_to_next_days)
            if next_day >= num_days:
                break
            eruption_days.append(int(next_day))
            current_day = next_day
        print(f"Generated eruption days: {eruption_days}")

    if not eruption_days:
        print("No eruptions occurred in the simulation period.")
        dates = pd.date_range("2010-01-01", periods=num_days)
        df = pd.DataFrame({"date": dates, "ash_mm_mean": np.zeros(num_days)})
        return df, []

    # --- 2. Precompute Grid Coordinates ---
    rows, cols = np.indices(dem.shape)
    
    # Reshape the coordinate lists back to the 2D shape of the DEM
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    xs = np.array(xs).reshape(dem.shape)
    ys = np.array(ys).reshape(dem.shape)
    
    dx = xs - volcano_x
    dy = ys - volcano_y
    distance_m = np.hypot(dx, dy)
    distance_km = distance_m / 1000.0
    theta_degrees = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    # --- 3. Simulate Each Eruption Event ---
    ash_mean_timeseries = np.zeros(num_days)
    for i, day in enumerate(eruption_days):
        if day >= num_days: continue
        print(f"Simulating eruption {i+1}/{len(eruption_days)} on day {day}...")
        Z_m, Z_gamma, Z_alpha, Z_beta = rng.standard_normal(4)

        M0 = 10**(10.36 + 0.3 * Z_m)
        alpha_eff = alpha0 * (M0**-0.25) * (10**(0.07 * Z_alpha))
        gamma_eff = gamma0 * (M0**0.46) * (10**(0.11 * Z_gamma))
        B = rng.beta(0.358, 0.317)
        phi_degrees = np.degrees(2 * np.pi * B)

        beta_u = -1.0
        while beta_u < 0:
            beta_u = 0.22 - (0.085 * alpha_eff) + (0.11 * Z_beta)
            if beta_u < 0: Z_beta = rng.standard_normal()

        grid_thickness_cm = calculate_thickness(
            distance_km, theta_degrees,
            gamma_eff, beta_u, phi_degrees, alpha_eff)
        grid_thickness_mm = grid_thickness_cm * 10.0

        grid_thickness_mm[np.isnan(dem)] = np.nan
        mean_ashfall_mm = np.nanmean(grid_thickness_mm)

        ash_mean_timeseries[day] += mean_ashfall_mm

    # --- 4. Build and return the final DataFrame ---
    dates = pd.date_range("2010-01-01", periods=num_days)
    df = pd.DataFrame({"date": dates, "ash_mm_mean": ash_mean_timeseries})
    return df, eruption_days
