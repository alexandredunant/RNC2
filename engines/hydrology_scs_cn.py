# engines/hydrology_scs_cn.py

import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
from typing import Dict, List, Optional, Any
from scipy import signal


# ──────────────── 1) SCS-CN Model Components ────────────────

class SCSCN_UH_Model:
    """
    SCS Curve Number model with Unit Hydrograph routing for a single catchment.
    Works on hourly timesteps.
    """
    def __init__(
        self,
        area_km2: float,
        cn: float = 70.0,
        length_km: float = 1.0,
        slope: float = 0.01,
        initial_abstraction_ratio: float = 0.2,
        channel_velocity_ms: float = 1.0,
        max_cn_increase: float = 25.0,
        k_ash: float = 0.5
    ):
        self.area_km2 = area_km2
        self.area_m2 = area_km2 * 1e6
        self.cn = cn
        self.length_km = length_km
        self.slope = max(slope, 0.001)
        self.ia_ratio = initial_abstraction_ratio
        self.channel_velocity_ms = channel_velocity_ms
        self.max_cn_increase = max_cn_increase
        self.k_ash = k_ash

        # SCS-CN parameters
        self.S_inch = (1000 / cn) - 10
        self.S_mm = self.S_inch * 25.4
        self.Ia_mm = self.ia_ratio * self.S_mm

        # Time of concentration (Giandotti formula)
        self.tc_hours = (4 * np.sqrt(area_km2) + 1.5 * length_km) / (0.8 * np.sqrt(self.slope * 1000))

        # Unit Hydrograph parameters (triangular)
        self.tp_hours = 0.6 * self.tc_hours
        self.tb_hours = 2.67 * self.tp_hours
        self.qp_per_mm = 0.208 * area_km2 / self.tp_hours

        self.uh = self._build_unit_hydrograph()

        self.cumulative_precip_mm = 0.0
        self.cumulative_runoff_mm = 0.0

    def _build_unit_hydrograph(self, dt_hours: float = 1.0) -> np.ndarray:
        """Build triangular unit hydrograph discretized at dt_hours intervals."""
        n_steps = int(np.ceil(self.tb_hours / dt_hours)) + 1
        uh = np.zeros(n_steps)

        for i in range(n_steps):
            t = i * dt_hours
            if t <= self.tp_hours:
                uh[i] = (t / self.tp_hours) * self.qp_per_mm
            elif t <= self.tb_hours:
                uh[i] = self.qp_per_mm * (self.tb_hours - t) / (self.tb_hours - self.tp_hours)
        return uh

    def _infiltration_reduction_ash(self, ash_depth_m: float) -> float:
        """Calculates the effective Curve Number based on ash depth."""
        ash_depth_mm = ash_depth_m * 1000
        cn_increase = self.max_cn_increase * (1 - np.exp(-self.k_ash * ash_depth_mm))
        effective_cn = min(98, self.cn + cn_increase)
        return effective_cn

    def compute_runoff_and_components(self, precip_mm: float, ash_depth_m: float = 0.0) -> Dict[str, float]:
        """Compute runoff using SCS-CN method for a single timestep."""
        cn_effective = self._infiltration_reduction_ash(ash_depth_m)
        S_mm_effective = (1000 / cn_effective - 10) * 25.4
        Ia_mm_effective = self.ia_ratio * S_mm_effective

        # Use event-based cumulative precipitation for runoff calculation
        self.cumulative_precip_mm += precip_mm

        if self.cumulative_precip_mm <= Ia_mm_effective:
            runoff_mm_total = 0.0
        else:
            excess = self.cumulative_precip_mm - Ia_mm_effective
            runoff_mm_total = (excess ** 2) / (excess + S_mm_effective)

        # Calculate the runoff generated in this specific step
        incremental_runoff = runoff_mm_total - self.cumulative_runoff_mm
        self.cumulative_runoff_mm = runoff_mm_total

        # Reset for next event if no rain
        if precip_mm == 0:
            self.cumulative_precip_mm = 0.0
            self.cumulative_runoff_mm = 0.0

        return {'runoff_mm': max(0.0, incremental_runoff)}


class CatchmentSCSCN:
    """Manages SCS-CN models for all catchments."""
    def __init__(self, catchment_df: pd.DataFrame, default_cn: float = 70.0, 
                 initial_abstraction_ratio: float = 0.2, 
                 max_cn_increase: float = 25.0, 
                 k_ash: float = 0.5):
        self.catchments: Dict[int, SCSCN_UH_Model] = {}
        self.ash_depths: Dict[int, float] = {}

        for cid, row in catchment_df.iterrows():
            area_km2 = row['catchment_area_m2'] / 1e6
            length_km = row.get('flow_length_km', np.sqrt(area_km2))
            slope = row.get('Slope (Gradient)', 0.01)
            cn = row.get('curve_number', default_cn)
            velocity = row.get('channel_velocity_ms', 1.0)
            ia_ratio = row.get('initial_abstraction_ratio', initial_abstraction_ratio)

            self.catchments[cid] = SCSCN_UH_Model(
                area_km2=area_km2, cn=cn, length_km=length_km, slope=slope,
                channel_velocity_ms=velocity, initial_abstraction_ratio=ia_ratio,
                max_cn_increase=max_cn_increase,
                k_ash=k_ash
            )
            self.ash_depths[cid] = 0.0

    def update_ash(self, ash_increment_m: float):
        """Update ash depth for all catchments."""
        for cid in self.ash_depths:
            self.ash_depths[cid] += ash_increment_m

    def compute_ash_wash(self, cid: int, incremental_runoff_mm: float, rho_ash: float = 1000.0, wash_efficiency: float = 0.01) -> float:
        """
        MODIFIED: Compute ash wash based on the incremental runoff from the hillslope.
        Erosion is now proportional to the volume of water running off in the current step.
        """
        model = self.catchments.get(cid)
        if not model or incremental_runoff_mm <= 0:
            return 0.0

        ash_depth_m = self.ash_depths.get(cid, 0.0)
        if ash_depth_m <= 0:
            return 0.0

        # Available ash mass in kg
        M_avail_kg = rho_ash * model.area_m2 * ash_depth_m
        
        # Simple erosion model: mass washed is proportional to runoff depth and efficiency.
        # We normalize by a reference runoff depth (e.g., 10mm) to keep wash_efficiency dimensionless.
        reference_runoff_depth_mm = 10.0
        fraction_to_wash = wash_efficiency * (incremental_runoff_mm / reference_runoff_depth_mm)

        M_wash_kg = M_avail_kg * fraction_to_wash
        M_wash_kg = min(M_avail_kg, M_wash_kg) # Cannot wash more than is available

        if M_avail_kg > 0:
            fraction_removed = M_wash_kg / M_avail_kg
            self.ash_depths[cid] *= (1 - fraction_removed)

        return M_wash_kg

    def update(self, precipitation_mm: float) -> tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Update all catchments for one timestep.
        MODIFIED: Now returns both the discharge series and the incremental runoff depths.
        """
        discharge_series = {}
        runoff_results = {}

        for cid, model in self.catchments.items():
            ash_m = self.ash_depths.get(cid, 0.0)
            result = model.compute_runoff_and_components(precipitation_mm, ash_m)
            runoff_mm = result['runoff_mm']
            runoff_results[cid] = runoff_mm

            if runoff_mm > 0:
                discharge_series[cid] = runoff_mm * model.uh
            else:
                discharge_series[cid] = np.zeros_like(model.uh)

        return discharge_series, runoff_results

# ──────────────── 2) D-Cascade Sediment Transport Classes ────────────────

class Reach:
    """Single river reach for sediment routing."""
    def __init__(
        self,
        id: int,
        downstream_id: Optional[int],
        length: float,
        slope: float,
        manning_n: float,
        width: float,
        storage: float = 0.0,
        transport_coefficient: float = 1e-5
    ):
        if length <= 0 or slope <= 0 or manning_n <= 0 or width <= 0:
            raise ValueError("Physical parameters must be positive")
        self.id = id
        self.downstream_id = downstream_id
        self.length = length
        self.slope = slope
        self.manning_n = manning_n
        self.width = width
        self.storage = {'ash': storage}
        self.discharge = 0.0
        self.metadata = {}
        self.transport_coefficient = transport_coefficient

    def add_sediment_mass(self, sediment_type: str, mass_kg: float) -> None:
        self.storage[sediment_type] = self.storage.get(sediment_type, 0.0) + mass_kg

    def compute_hydraulic_radius(self) -> float:
        return (self.discharge / self.width)**0.375 if self.discharge > 0 else 0.01

    def compute_transport_capacity(self) -> float:
        ρ, g = 1000.0, 9.81
        R = self.compute_hydraulic_radius()
        shear = ρ * g * R * self.slope
        return (shear**1.5) * self.width * self.length * self.transport_coefficient

    def step(self, incoming_flux_kg: float, dt: float = 1.0) -> float:
        self.add_sediment_mass('ash', incoming_flux_kg)
        avail = self.storage['ash']
        if avail > 0:
            cap = self.compute_transport_capacity()
            if avail <= cap:
                exported = avail
                self.storage['ash'] = 0.0
            else:
                exported = cap
                self.storage['ash'] = avail - cap
            return exported
        return 0.0


class DCascade:
    """Network-wide sediment routing."""
    def __init__(self, reaches: Dict[int, Reach]):
        self.reaches = reaches
        self.graph = nx.DiGraph()
        for rid in reaches: self.graph.add_node(rid)
        for rid, rc in reaches.items():
            if rc.downstream_id is not None: self.graph.add_edge(rid, rc.downstream_id)
        self.topo_order = list(nx.topological_sort(self.graph))

    def _upstream_ids(self, rid: int) -> List[int]:
        return list(self.graph.predecessors(rid))

    def step(self, dt: float = 1.0) -> Dict[int, float]:
        exports: Dict[int, float] = {}
        for rid in self.topo_order:
            inc = sum(exports.get(u, 0.0) for u in self._upstream_ids(rid))
            exports[rid] = self.reaches[rid].step(inc, dt)
        return exports


def build_network(edges: gpd.GeoDataFrame, transport_coefficient: float = 1e-5) -> DCascade:
    reaches: Dict[int, Reach] = {}
    for _, row in edges.iterrows():
        rid, ds = int(row['reach_id']), row['downstream_id']
        reaches[rid] = Reach(
            id=rid,
            downstream_id=int(ds) if pd.notna(ds) else None,
            length=float(row['length']),
            slope=float(row['Slope (Gradient)']),
            manning_n=float(row['manning_n']),
            width=float(row['width']),
            storage=float(row.get('initial_storage_ash_kg', 0.0)),
            transport_coefficient=transport_coefficient
        )
        reaches[rid].metadata['A_sub'] = row['A_sub']
    return DCascade(reaches)

# ──────────────── 3) Network Routing with Lag and Sediment ────────────────

class NetworkRouterWithSediment:
    """Routes discharge through the network using pure lag (translation) and handles sediment transport."""
    def __init__(self, edges_gdf: gpd.GeoDataFrame, dt_hours: float = 1.0, transport_coefficient: float = 1e-5):
        self.dt_hours = dt_hours
        self.graph = nx.DiGraph()
        for _, row in edges_gdf.iterrows(): self.graph.add_node(int(row['reach_id']))
        for _, row in edges_gdf.iterrows():
            if pd.notna(row['downstream_id']) and row['downstream_id'] > 0: self.graph.add_edge(int(row['reach_id']), int(row['downstream_id']))
        self.topo_order = list(nx.topological_sort(self.graph))

        self.reach_lags = {}
        for _, row in edges_gdf.iterrows():
            velocity_ms = row.get('channel_velocity_ms', 1.0)
            lag_hours = (row['length'] / 1000) / (velocity_ms * 3.6)
            self.reach_lags[int(row['reach_id'])] = max(1, int(np.round(lag_hours / dt_hours)))

        max_lag = max(self.reach_lags.values()) if self.reach_lags else 10
        self.buffer_size = max_lag + 200
        self.discharge_buffers = {rid: np.zeros(self.buffer_size) for rid in self.topo_order}
        self.sediment_model = build_network(edges_gdf, transport_coefficient=transport_coefficient)

    def add_lateral_inflow(self, reach_id: int, discharge_array: np.ndarray, current_step: int):
        n = len(discharge_array)
        start_idx = current_step % self.buffer_size
        for i in range(n):
            idx = (start_idx + i) % self.buffer_size
            self.discharge_buffers[reach_id][idx] += discharge_array[i]

    def add_sediment_input(self, reach_id: int, sediment_mass_kg: float):
        if reach_id in self.sediment_model.reaches:
            self.sediment_model.reaches[reach_id].add_sediment_mass('ash', sediment_mass_kg)

    def route_step(self, current_step: int) -> Dict[str, Any]:
        """Route one timestep through the network."""
        current_discharge = {}
        for rid in self.topo_order:
            idx = current_step % self.buffer_size
            q_current = self.discharge_buffers[rid][idx]
            current_discharge[rid] = q_current
            if rid in self.sediment_model.reaches: self.sediment_model.reaches[rid].discharge = q_current
            self.discharge_buffers[rid][idx] = 0.0
            if rid in self.graph:
                for ds_rid in self.graph.successors(rid):
                    future_idx = (current_step + self.reach_lags[rid]) % self.buffer_size
                    self.discharge_buffers[ds_rid][future_idx] += q_current

        sediment_exports = self.sediment_model.step(dt=self.dt_hours * 3600)
        return {'discharge': current_discharge, 'sediment_flux': sediment_exports}

    def get_sediment_state(self) -> Dict[int, float]:
        return {rid: reach.storage.get('ash', 0.0) for rid, reach in self.sediment_model.reaches.items()}

# ──────────────── 4) Hourly Data Disaggregation ────────────────

def disaggregate_daily_to_hourly(daily_df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    hourly_data = []
    for _, row in daily_df.iterrows():
        date, daily_value = row['date'], row[value_col]
        hourly_value = daily_value / 24.0
        for hour in range(24):
            hourly_data.append({'datetime': pd.Timestamp(date) + pd.Timedelta(hours=hour), 'date': date, 'hour': hour, value_col: hourly_value})
    return pd.DataFrame(hourly_data)

# ──────────────── 5) Main Simulation Function ────────────────

def run_scscn_uh_cascade(
    edges_gdf: gpd.GeoDataFrame, catch_df: pd.DataFrame, precip_df: pd.DataFrame, ash_df: pd.DataFrame,
    default_cn: float = 70.0, channel_velocity_ms: float = 1.0, initial_abstraction_ratio: float = 0.2,
    dt_hours: float = 1.0, rho_ash: float = 1000.0, wash_efficiency: float = 0.01,
    transport_coefficient: float = 1e-5, ash_cn_multiplier: float = 1.0,
    max_cn_increase: float = 25.0,
    k_ash: float = 0.5,
    **kwargs
) -> Dict[str, Any]:

    if 'channel_velocity_ms' not in edges_gdf.columns: edges_gdf['channel_velocity_ms'] = channel_velocity_ms
    if 'channel_velocity_ms' not in catch_df.columns: catch_df['channel_velocity_ms'] = channel_velocity_ms

    precip_hourly = disaggregate_daily_to_hourly(precip_df, 'precip_mm')
    ash_hourly = disaggregate_daily_to_hourly(ash_df, 'ash_mm_mean')
    hourly_df = precip_hourly.merge(ash_hourly[['datetime', 'ash_mm_mean']], on='datetime', how='left').fillna(0)

    scscn = CatchmentSCSCN(catch_df, default_cn=default_cn, 
                           initial_abstraction_ratio=initial_abstraction_ratio,
                           max_cn_increase=max_cn_increase,
                           k_ash=k_ash)
    router = NetworkRouterWithSediment(edges_gdf, dt_hours=dt_hours, 
                                       transport_coefficient=transport_coefficient)
    catch_to_reach = edges_gdf.groupby('catchment_id')['reach_id'].first().to_dict()

    discharge_ts, network_snapshots, hillslope_ash_depth_ts, cumulative_ash_wash_ts, ash_discharge_ts = [], [], [], [], []
    cumulative_wash_mass = 0.0

    print(f"Running {len(hourly_df)} hourly timesteps with sediment transport...")
    for step, (idx, row) in enumerate(tqdm(hourly_df.iterrows(), total=len(hourly_df))):
        ash_m = row['ash_mm_mean'] / 1000.0
        scscn.update_ash(ash_m)

        # MODIFICATION: Get both discharge hydrographs and the incremental runoff depths
        discharge_series, incremental_runoff = scscn.update(precipitation_mm=row['precip_mm'])

        for cid, q_series in discharge_series.items():
            if cid in catch_to_reach:
                router.add_lateral_inflow(catch_to_reach[cid], q_series, step)

        # MODIFICATION: Drive ash wash with the calculated incremental runoff for each catchment
        total_ash_washed_this_step = 0
        for cid, runoff_mm in incremental_runoff.items():
            if runoff_mm > 0 and cid in catch_to_reach:
                rid = catch_to_reach[cid]
                ash_mass = scscn.compute_ash_wash(cid, runoff_mm, rho_ash, wash_efficiency=wash_efficiency)
                if ash_mass > 0:
                    router.add_sediment_input(rid, ash_mass)
                    total_ash_washed_this_step += ash_mass

        routing_results = router.route_step(step)
        current_reach_discharge = routing_results['discharge']
        sediment_exports = routing_results['sediment_flux']

        # --- Record diagnostic data ---
        mean_hillslope_depth_mm = np.mean(list(scscn.ash_depths.values())) * 1000.0
        hillslope_ash_depth_ts.append({'datetime': row['datetime'], 'mean_ash_depth_mm': mean_hillslope_depth_mm})

        cumulative_wash_mass += total_ash_washed_this_step
        cumulative_ash_wash_ts.append({'datetime': row['datetime'], 'cumulative_ash_wash_kg': cumulative_wash_mass})

        outlet_reaches = [r for r in router.topo_order if not list(router.graph.successors(r))]
        outlet_q = sum(current_reach_discharge.get(r, 0) for r in outlet_reaches) if outlet_reaches else sum(current_reach_discharge.values())
        discharge_ts.append({'datetime': row['datetime'], 'discharge_m3s': outlet_q})

        outlet_ash_flux_kg_per_hour = sum(sediment_exports.get(r, 0.0) for r in outlet_reaches)
        ash_discharge_ts.append({'datetime': row['datetime'], 'ash_flux_kg_s': outlet_ash_flux_kg_per_hour / 3600.0})

        if step % 24 == 0:
            sediment_state = router.get_sediment_state()
            reach_ids = list(current_reach_discharge.keys())
            snapshot_data = {
                'reach_id': list(current_reach_discharge.keys()),
                'discharge_m3s': list(current_reach_discharge.values()),
                'ash_storage': [sediment_state.get(rid, 0) for rid in current_reach_discharge.keys()],
                'ash_flow_kgs': [sediment_exports.get(rid, 0) / 3600.0 for rid in reach_ids],
                'time_step': step, 
                'datetime': row['datetime']
            }
            network_snapshots.append(pd.DataFrame(snapshot_data))

    return {
        'discharge_ts': pd.DataFrame(discharge_ts),
        'network_snapshots': network_snapshots,
        'static_network_gdf': edges_gdf,
        'scscn_model': scscn, 'router': router,
        'hourly_data': hourly_df,
        'sediment_model': router.sediment_model,
        'hillslope_ash_depth_ts': pd.DataFrame(hillslope_ash_depth_ts),
        'cumulative_ash_wash_ts': pd.DataFrame(cumulative_ash_wash_ts),
        'ash_discharge_ts': pd.DataFrame(ash_discharge_ts)
    }
