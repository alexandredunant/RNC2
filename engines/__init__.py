# engines/__init__.py

from .precip      import generate as generate_precip
from .ashfall     import generate as generate_ashfall
from .ashfall import load_dem_from_pygmt, save_dem_to_file, load_dem_from_file
from .rivernet    import build_from_dem
from .hydrology   import run_nlrm_cascade
from .visualize   import timeseries, network_animation

# Optional: also export the SCS-CN specific function if needed
try:
    from .hydrology_scs_cn import run_scscn_uh_cascade
except ImportError:
    run_scscn_uh_cascade = None

__all__ = [
    "generate_precip",
    "generate_ashfall",
    "load_dem_from_pygmt",
    "save_dem_to_file",
    "load_dem_from_file",
    "build_from_dem",
    "run_nlrm_cascade",
    "timeseries",
    "network_animation",
]

if run_scscn_uh_cascade is not None:
    __all__.append("run_scscn_uh_cascade")