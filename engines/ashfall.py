# engines/ashfall.py

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
import os

# --- Helper function for eruption timing ---
def generate_timeline(num_days: int, eruption_days: list[int]) -> np.ndarray:
    timeline = np.zeros(num_days, dtype=int)
    for d in eruption_days:
        if 0 <= d < num_days: timeline[d] = 1
    return timeline

# --- New Thickness Calculation Function (updated to accept parameters) ---
def calculate_thickness(
    distance_km: np.ndarray,
    theta_degrees: np.ndarray,
    gamma: float,      # Eruption size proxy
    beta_u: float,     # Wind effect parameter
    phi_degrees: float,# Main dispersal axis
    alpha: float       # Distance decay exponent
) -> np.ndarray:
    """
    Calculates tephra thickness using the Gonzalez-Mellado & De la Cruz-Reyna (2010) model.
    Returns thickness in cm.
    """
    # Convert angles to radians for numpy's trigonometric functions
    theta_rad = np.radians(theta_degrees)
    phi_rad = np.radians(phi_degrees)

    # Calculate the cosine term for the wind effect
    cos_term = np.cos(theta_rad - phi_rad)

    # Calculate the two main terms of the model
    wind_term = np.exp(-beta_u * distance_km * (1 - cos_term))

    # Use a small epsilon to prevent division by zero at the vent (r=0)
    decay_term = (distance_km + 1e-9)**(-alpha)

    # Final thickness in cm
    T_cm = gamma * wind_term * decay_term

    return T_cm

# --- Utility functions for loading DEM data (unchanged) ---
def load_dem_from_pygmt(region, resolution="01s"):
    try:
        import pygmt
        grid = pygmt.datasets.load_earth_relief(resolution=resolution, region=region)
        data, transform, crs = grid.data, from_origin(grid.coords["lon"].min(), grid.coords["lat"].max(), grid.coords["lon"][1] - grid.coords["lon"][0], grid.coords["lat"][0] - grid.coords["lat"][1]), "EPSG:4326"
        return data, transform, crs
    except ImportError: return None, None, None

def load_dem_from_file(dem_path):
    with rasterio.open(dem_path) as src:
        elev, transform, crs = src.read(1).astype(float), src.transform, src.crs
        if src.nodata is not None: elev[elev == src.nodata] = np.nan
        return elev, transform, crs

def save_dem_to_file(data, transform, crs, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with rasterio.open(output_path, "w", driver="GTiff", height=data.shape[0], width=data.shape[1], count=1, dtype=data.dtype, crs=crs, transform=transform) as dst:
        dst.write(data, 1)

# ──────────────── Public API (updated to use config parameters) ────────────────
def generate(
    *,  # enforce keyword-only
    num_days: int,
    seed: int,
    dem: np.ndarray,
    transform: rasterio.Affine,
    crs: rasterio.crs.CRS,
    volcano_x: float,
    volcano_y: float,
    # Parameters from config file
    M0: float,
    wind_mu: float,
    wind_sigma: float,
    gamma0: float,
    eta0: float,
    **kwargs # to absorb other unused parameters
) -> pd.DataFrame:
    """
    Generate tephra (ash) time series based on model parameters passed from the config.
    """
    rng = np.random.default_rng(seed)

    # --- Use config parameters instead of hardcoded values ---
    # Simple scaling law for eruption size proxy
    GAMMA = gamma0 * (10**M0)
    # Use wind direction from config
    PHI_DEGREES = rng.normal(wind_mu, wind_sigma)
    # Simple scaling for distance decay
    ALPHA = eta0 / (M0 + 1)
    # Wind effect can be kept constant or scaled as well
    BETA_U = 1.48

    print(f"Generating ashfall with dynamic params: GAMMA={GAMMA:.2f}, PHI={PHI_DEGREES:.1f}, ALPHA={ALPHA:.3f}")

    # Eruption timing model
    eruption_days = [20, 30, 200]
    shape_b, scale_a = 0.213, 39.8
    while True:
        delta = scale_a * rng.weibull(shape_b)
        next_day = eruption_days[-1] + np.ceil(delta)
        if next_day >= num_days: break
        eruption_days.append(int(next_day))
    timeline = generate_timeline(num_days, eruption_days)

    # Precompute grid coordinates, distance, and azimuth
    rows, cols = np.indices(dem.shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset='center')
    dx = np.array(xs) - volcano_x
    dy = np.array(ys) - volcano_y

    distance_m = np.hypot(dx, dy)
    distance_km = distance_m / 1000.0

    theta_degrees = (np.degrees(np.arctan2(dy, dx)) + 360) % 360

    # Calculate thickness using the dynamic parameters
    grid_thickness_cm = calculate_thickness(
        distance_km, theta_degrees,
        gamma=GAMMA,
        beta_u=BETA_U,
        phi_degrees=PHI_DEGREES,
        alpha=ALPHA
    )
    grid_thickness_mm = grid_thickness_cm * 10.0 # Convert to mm
    max_thickness_mm = np.nanmax(grid_thickness_mm)

    # Create the time series
    ash_mean = np.zeros(num_days)
    for day in eruption_days:
        if day < num_days:
            ash_mean[day] = max_thickness_mm

    # Build and return the final DataFrame
    dates = pd.date_range("2010-01-01", periods=num_days)
    df = pd.DataFrame({"date": dates, "ash_mm_mean": ash_mean})

    return df