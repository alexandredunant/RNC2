![RNC2 project](https://github.com/37stu37/rnc2_scripts/blob/main/project%20image.png)

-----

> Learn more about the [Resilience to Nature's Challenges National Science Challenge](https://resiliencechallenge.nz/).

-----

## Table of Contents

  - [Overview](https://www.google.com/search?q=%23overview)
  - [Project Structure](https://www.google.com/search?q=%23project-structure)
  - [Usage](https://www.google.com/search?q=%23usage)
  - [Key Scripts and Engines](https://www.google.com/search?q=%23key-scripts-and-engines)
  - [Data](https://www.google.com/search?q=%23data)
  - [Publications](https://www.google.com/search?q=%23publications)

-----

## Overview

This repository contains a set of Python scripts for simulating the hydrological and sedimentological response of a river catchment to rainfall and volcanic ashfall. The primary model is a hydrology and sediment transport model that can be run with an hourly SCS-CN (Soil Conservation Service Curve Number) Unit Hydrograph model. The model is designed to simulate the effects of volcanic eruptions on river systems, including ash deposition, wash-off from hillslopes, and transport through the river network.

## Project Structure

The repository is organized into the following key directories and files:

```
.
├── engines/
│   ├── __init__.py
│   ├── ashfall.py
│   ├── hydrology.py
│   ├── hydrology_scs_cn.py
│   ├── precip.py
│   ├── rivernet.py
│   └── visualize.py
├── outputs/
│   └── ... (simulation results)
├── run_simulation_scscn.py
├── simulate_case.py
└── README.md
```

  - **`run_simulation_scscn.py`**: The main script for running simulations.
  - **`simulate_case.py`**: A helper script that sets up and runs a single simulation case.
  - **`engines/`**: A directory containing the core modeling components.
  - **`outputs/`**: The directory where simulation results are saved.

## Usage

To run a set of simulations, you can execute the `run_simulation_scscn.py` script. This script allows you to define a range of parameter sets to be simulated. Each parameter set is defined as a dictionary within the `parameter_sets` list.

```python
# Example of a parameter set in run_simulation_scscn.py
parameter_sets.append(dict(
    tag               = f"CN70_V0.5_IA0.05_S0.2_R2500_M1.0_WE0.2_TC0.0005_AM10.0",
    hydro_model       = "scscn",
    curve_number      = 70,
    channel_velocity_ms = 0.5,
    # ... other parameters
))
```

When you run the script, it will iterate through each parameter set, run the simulation, and save the results to a corresponding subdirectory within the `outputs` directory.

## Key Scripts and Engines

### Main Scripts

  - **`run_simulation_scscn.py`**: This is the main entry point for running simulations. It defines various parameter sets to test different scenarios and then calls `simulate_case.py` for each set.
  - **`simulate_case.py`**: This script handles a single simulation case. It takes a configuration dictionary, loads or generates the necessary data (DEM, precipitation, ashfall), builds the river network, runs the selected hydrology model, and saves the results.

### Engines

The `engines` directory contains the core components of the model:

  - **`hydrology_scs_cn.py`**: Implements the SCS Curve Number model with a Unit Hydrograph for hourly hydrological routing. This includes calculations for runoff, infiltration, and the effects of ash on the curve number.
  - **`hydrology.py`**: Contains the `run_nlrm_cascade` function which can run either the Non-Linear Reservoir Model (NLRM) or the SCS-CN model. It also includes a 'Cascade` model for sediment transport in the river network.
  - **`ashfall.py`**: Generates tephra (ash) thickness and distribution based on eruption parameters and a physical model.
  - **`precip.py`**: Generates daily precipitation time series using a gamma distribution and seasonal adjustments.
  - **`rivernet.py`**: Builds the river network and catchments from a Digital Elevation Model (DEM) using `pyflwdir`.
  - **`visualize.py`**: Provides functions for creating plots and animations of the simulation results, including timeseries plots and network animations.

## Data

The GIS input https://www.google.com/search?q=data required for the scripts are located in the `data` folder (not included in this repository). This https://www.google.com/search?q=data includes geospatial information about catchments, river networks, and other relevant parameters necessary for the simulations. Specifically, the model requires a DEM, which can be provided as a GeoTIFF file. If a local DEM is not found, the `simulate_case.py` script can download one using `pygmt`.

-----
