### README.md

![RNC2 project](https://github.com/37stu37/rnc2_scripts/blob/main/project%20image.png)

-----

> Learn more about the [Resilience to Nature's Challenges National Science Challenge](https://resiliencechallenge.nz/).

-----

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Key Scripts and Engines](#key-scripts-and-engines)
- [Data](#data)
- [Citation](#citation)

-----

## Overview

This repository contains a set of Python scripts for simulating the hydrological and sedimentological response of a river catchment to rainfall and volcanic ashfall. The primary model is a hydrology and sediment transport model that can be run with an hourly SCS-CN (Soil Conservation Service Curve Number) Unit Hydrograph model. The model is designed to simulate the effects of volcanic eruptions on river systems, including ash deposition, wash-off from hillslopes, and transport through the river network.

## Project Structure

The repository is organized into the following key directories and files:

```

.
├── engines/
│   ├── **init**.py
│   ├── ashfall.py
│   ├── hydrology.py
│   ├── hydrology\_scs\_cn.py
│   ├── precip.py
│   ├── rivernet.py
│   └── visualize.py
├── outputs/
│   └── ... (simulation results)
├── run\_simulation.py
├── run\_simulation\_comparison.py
├── simulate\_case.py
└── README.md

````

- **`run_simulation.py`**: The main script for running a single simulation with a defined set of parameters.
- **`run_simulation_comparison.py`**: A script to run and compare two scenarios, such as with and without ashfall.
- **`simulate_case.py`**: A helper script that sets up and runs a single simulation case.
- **`engines/`**: A directory containing the core modeling components.
- **`outputs/`**: The directory where simulation results are saved.

## Usage

To run a set of simulations, you can execute the `run_simulation.py` script. This script allows you to define a parameter set to be simulated. The parameter set is defined as a dictionary.

```python
# Example of a parameter set in run_simulation.py
cfg = dict(
    tag               = "TEST_180_eruption_new_hydo_20year",
    hydro_model       = "scscn",
    curve_number      = 70,
    # ... other parameters
)
````

When you run the script, it will execute the simulation and save the results to a corresponding subdirectory within the `outputs` directory.

To compare two different scenarios, you can use the `run_simulation_comparison.py` script, which runs two simulations back-to-back and generates comparative plots.

## Key Scripts and Engines

### Main Scripts

  - **`run_simulation.py`**: This is the main entry point for running a single simulation. It defines a parameter set to test a specific scenario and then calls `simulate_case.py`.
  - **`run_simulation_comparison.py`**: This script is designed to compare two different scenarios, for example, with and without an ashfall event. It defines two separate configurations and runs them sequentially, generating a suite of comparative plots.
  - **`simulate_case.py`**: This script handles a single simulation case. It takes a configuration dictionary, loads or generates the necessary data (DEM, precipitation, ashfall), builds the river network, runs the selected hydrology model, and saves the results.

### Engines

The `engines` directory contains the core components of the model:

  - **`hydrology_scs_cn.py`**: Implements the SCS Curve Number model with a Unit Hydrograph for hourly hydrological routing. This includes calculations for runoff, infiltration, and the effects of ash on the curve number.
  - **`hydrology.py`**: Contains the `run_nlrm_cascade` function which can run the SCS-CN model. It also includes a 'Cascade\` model for sediment transport in the river network.
  - **`ashfall.py`**: Generates tephra (ash) thickness and distribution based on eruption parameters and a physical model.
  - **`precip.py`**: Generates daily precipitation time series using a gamma distribution and seasonal adjustments.
  - **`rivernet.py`**: Builds the river network and catchments from a Digital Elevation Model (DEM) using `pyflwdir`.
  - **`visualize.py`**: Provides functions for creating plots and animations of the simulation results, including timeseries plots and network animations.

## Data

The GIS input data required for the scripts are located in the `data` folder (not included in this repository). Specifically, the model requires a DEM, which can be provided as a GeoTIFF file.

## Citation

```
N/A


```
```
