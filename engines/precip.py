import numpy as np
import pandas as pd

def generate(**cfg):
    """Return DataFrame(date, precip_mm) using explicit numbers in cfg."""
    rng   = np.random.default_rng(cfg['seed'])
    mu    = cfg['mean_annual_precip'] / 365
    depth = rng.gamma(shape=cfg['precip_sigma'], scale=mu/cfg['precip_sigma'], size=cfg['num_days'])
    depth *= (rng.random(cfg['num_days']) < cfg['rain_prob'])

    t     = np.arange(cfg['num_days'])
    depth *= 1 + cfg['season_strength'] * np.sin(2 * np.pi * t / 365)

    return pd.DataFrame({
        'date': pd.date_range('2010-01-01', periods=cfg['num_days']),
        'precip_mm': depth
    })
