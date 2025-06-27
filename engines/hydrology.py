# engines/hydrology.py

import math
import numpy as np
import pandas as pd
import geopandas as gpd
import networkx as nx
from tqdm import tqdm
from typing import Dict, List, Optional, Union, Any

# Import the new SCS-CN model
from .hydrology_scs_cn import run_scscn_uh_cascade

# ──────────────── 1) Non-Linear Reservoir Model ────────────────

class NonLinearReservoir:
    """
    A non-linear reservoir for soil moisture and runoff.
    """
    def __init__(
        self,
        area: float,
        k: float = 0.03,
        n: float = 1.5,
        initial_storage: float = 0.0,
        max_storage: float = 1e9,
        baseflow_coeff: float = 0.001,
        interflow_coeff: float = 0.01,
        soil_moisture_capacity: float = 100.0,
        initial_soil_moisture: float = 50.0
    ):
        self.area = area
        self.k = k
        self.n = n
        self.storage = initial_storage
        self.max_storage = max_storage
        self.baseflow_coeff = baseflow_coeff
        self.interflow_coeff = interflow_coeff
        self.soil_moisture_capacity = soil_moisture_capacity
        self.soil_moisture = initial_soil_moisture

        self.surface_flow = 0.0
        self.interflow = 0.0
        self.baseflow = 0.0
        self.nonlinear_flow = 0.0
        self.total_outflow = 0.0

    def _infiltration_reduction(self, ash_depth_m: float) -> float:
        """Reduce infiltration linearly with ash depth (max 90%)."""
        return 1.0 - min(0.9, ash_depth_m * 100)

    def update(
        self,
        precipitation_mm: float,
        evapotranspiration_mm_per_day: float,
        ash_depth_m: float,
        dt: float = 86400.0
    ) -> float:
        """
        Advance one timestep: soil-moisture accounting → flows → storage update.
        Returns total outflow (m³/s).
        """
        # 1) Soil moisture + ET
        self.soil_moisture += precipitation_mm
        et = evapotranspiration_mm_per_day * (dt / 86400.0)
        self.soil_moisture = max(0.0, self.soil_moisture - et)

        # 2) Infiltration capacity
        infil_red = self._infiltration_reduction(ash_depth_m)
        cap_mm = self.soil_moisture_capacity * infil_red

        # 3) Surface runoff if soil moisture > capacity
        if self.soil_moisture > cap_mm:
            excess = self.soil_moisture - cap_mm
            self.soil_moisture = cap_mm
            vol_excess = (excess / 1000.0) * self.area
            self.surface_flow = vol_excess / dt
        else:
            self.surface_flow = 0.0

        # 4) Infiltrated volume adds to storage
        precip_vol = (precipitation_mm / 1000.0) * self.area
        infil_vol = precip_vol - (self.surface_flow * dt)
        self.storage += infil_vol

        # 5) Interflow
        inter_mm = self.interflow_coeff * self.soil_moisture
        self.soil_moisture = max(0.0, self.soil_moisture - inter_mm)
        inter_vol = (inter_mm / 1000.0) * self.area
        self.interflow = inter_vol / dt

        # 6) Baseflow
        depth = self.storage / self.area
        self.baseflow = self.baseflow_coeff * depth * self.area

        # 7) Non-linear reservoir release
        if self.storage > 0:
            self.nonlinear_flow = self.k * (depth ** self.n) * self.area
        else:
            self.nonlinear_flow = 0.0

        # 8) Update storage
        remove = (self.interflow + self.baseflow + self.nonlinear_flow) * dt
        self.storage = max(0.0, self.storage - remove)

        # 9) Spill if above max_storage
        if self.storage > self.max_storage:
            spill = self.storage - self.max_storage
            self.storage = self.max_storage
            self.surface_flow += spill / dt

        # 10) Total outflow
        self.total_outflow = (
            self.surface_flow +
            self.interflow +
            self.baseflow +
            self.nonlinear_flow
        )
        return float(self.total_outflow)

    def get_flow_components(self) -> Dict[str, float]:
        """Return individual flow components."""
        return {
            'surface_flow': self.surface_flow,
            'interflow':     self.interflow,
            'baseflow':      self.baseflow,
            'nonlinear_flow':self.nonlinear_flow,
            'total_outflow': self.total_outflow,
            'storage':       self.storage,
            'soil_moisture_mm': self.soil_moisture
        }


class CatchmentNLRM:
    """
    Manages one NonLinearReservoir per catchment.
    """
    def __init__(self, catchment_df: pd.DataFrame):
        # identify shared catchments
        catchment_df["A_sub"] = catchment_df["catchment_area_m2"]
        shared = catchment_df.groupby('catchment_id')['A_sub'].sum()
        own    = catchment_df.groupby('catchment_id')['A_sub'].first()
        self.shared_catchments = shared[shared > own].index.tolist()

        # initialize reservoirs
        self.catchments: Dict[int, NonLinearReservoir] = {}
        for cid, row in catchment_df.iterrows():
            self.catchments[cid] = NonLinearReservoir(
                area=row['A_sub'],
                k=row.get('k', 0.03),
                n=row.get('n', 1.5),
                baseflow_coeff=row.get('baseflow_coeff', 0.001),
                interflow_coeff=row.get('interflow_coeff', 0.01),
                soil_moisture_capacity=row.get('soil_moisture_capacity', 100.0),
                initial_soil_moisture=row.get('initial_soil_moisture', 50.0)
            )

    def update(
        self,
        precipitation_m: float,
        evapotranspiration: float,
        ash_depths: Dict[int, float],
        dt: float = 86400.0
    ) -> Dict[int, float]:
        """
        Advance each catchment one step.
        precipitation_m in metres.
        Returns catchment discharges (m³/s).
        """
        out: Dict[int, float] = {}
        for cid, res in self.catchments.items():
            precip_mm = precipitation_m * 1000.0
            ash_m     = ash_depths.get(cid, 0.0)
            Q         = res.update(precip_mm, evapotranspiration, ash_m, dt)
            out[cid]  = Q
        return out

    def get_flow_components(self) -> Dict[int, Dict[str, float]]:
        """Get all flow components per catchment."""
        return {cid: res.get_flow_components() for cid, res in self.catchments.items()}


# ──────────────── 2) D-Cascade Model ────────────────

class Reach:
    """
    Single river reach for sediment routing.
    """
    def __init__(
        self,
        id: int,
        downstream_id: Optional[int],
        length: float,
        slope: float,
        manning_n: float,
        width: float,
        storage: float = 0.0
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
        self.deposit_gsd = None
        self.outflow_gsd = None

    def add_sediment_mass(self, sediment_type: str, mass_kg: float) -> None:
        self.storage[sediment_type] = self.storage.get(sediment_type, 0.0) + mass_kg

    def compute_hydraulic_radius(self) -> float:
        return (self.discharge / self.width)**0.375 if self.discharge > 0 else 0.01

    def compute_transport_capacity(self) -> float:
        ρ, g = 1000.0, 9.81
        R = self.compute_hydraulic_radius()
        shear = ρ * g * R * self.slope
        return (shear**1.5) * self.width * self.length * 1e-3

    def step(self, incoming_flux_kg: float, dt: float = 1.0) -> float:
        self.add_sediment_mass('ash', incoming_flux_kg)
        avail = self.storage['ash']
        cap   = self.compute_transport_capacity()
        if avail <= cap:
            exported = avail
            self.storage['ash'] = 0.0
        else:
            exported = cap
            self.storage['ash'] = avail - cap
        return exported


class DCascade:
    """
    Network‐wide sediment routing.
    """
    def __init__(self, reaches: Dict[int, Reach]):
        self.reaches = reaches
        self.graph = nx.DiGraph()
        for rid in reaches:
            self.graph.add_node(rid)
        for rid, rc in reaches.items():
            if rc.downstream_id is not None:
                self.graph.add_edge(rid, rc.downstream_id)
        self.topo_order = list(nx.topological_sort(self.graph))

    def _upstream_ids(self, rid: int) -> List[int]:
        return list(self.graph.predecessors(rid))

    def step(self, dt: float = 1.0) -> Dict[int, float]:
        exports: Dict[int, float] = {}
        for rid in self.topo_order:
            inc = sum(exports.get(u, 0.0) for u in self._upstream_ids(rid))
            exports[rid] = self.reaches[rid].step(inc, dt)
        return exports


def build_network(edges: gpd.GeoDataFrame) -> DCascade:
    """
    Construct a DCascade model from a GeoDataFrame of reaches.
    """
    reaches: Dict[int, Reach] = {}
    for _, row in edges.iterrows():
        rid = int(row['reach_id'])
        ds  = row['downstream_id']
        reaches[rid] = Reach(
            id=rid,
            downstream_id=int(ds) if pd.notna(ds) else None,
            length=float(row['length']),
            slope=float(row['Slope (Gradient)']),
            manning_n=float(row['manning_n']),
            width=float(row['width']),
            storage=float(row.get('initial_storage_ash_kg', 0.0))
        )
        reaches[rid].metadata['A_sub'] = row['A_sub']
    return DCascade(reaches)


# ──────────────── 3) Public API ────────────────

def run_nlrm_cascade(
    edges_gdf: gpd.GeoDataFrame,
    catch_df: pd.DataFrame,
    precip_df: pd.DataFrame,
    ash_df: pd.DataFrame,
    k: float,
    n: float,
    baseflow_coeff: float,
    evapotranspiration: float = 3.0,
    rho_ash: float = 1000.0,
    model_type: str = "nlrm",  # New parameter to select model type
    wash_efficiency: float = 0.01,
    transport_coefficient: float = 1e-5,
    max_cn_increase: float = 25.0,
    k_ash: float = 0.5
    # ash_cn_multiplier: float = 1.0
) -> Dict[str, Any]:
    """
    Run hydrology + sediment cascade model.

    Parameters:
    - model_type: "nlrm" for Non-Linear Reservoir Model (daily)
                  "scscn" for SCS-CN with Unit Hydrograph (hourly)
    """

    if model_type == "scscn":
        # Use the new SCS-CN model with hourly timesteps
        return run_scscn_uh_cascade(
            edges_gdf=edges_gdf,
            catch_df=catch_df,
            precip_df=precip_df,
            ash_df=ash_df,
            default_cn=catch_df.get('curve_number', {}).get(0, 70.0),  # Get CN from first catchment or default
            channel_velocity_ms=catch_df.get('channel_velocity_ms', {}).get(0, 1.0),  # Get velocity from first catchment or default
            dt_hours=1.0,
            rho_ash=rho_ash,
            wash_efficiency=wash_efficiency,
            transport_coefficient=transport_coefficient,
            max_cn_increase=max_cn_increase,
            k_ash=k_ash
            # ash_cn_multiplier=ash_cn_multiplier
        )

    # Otherwise use the original NLRM model (daily)
    # 1) set hydro params
    for cid in catch_df.index:
        catch_df.at[cid, 'k']              = k
        catch_df.at[cid, 'n']              = n
        catch_df.at[cid, 'baseflow_coeff'] = baseflow_coeff

    # 2) merge precipitation & ash
    df = precip_df.merge(ash_df, on='date')
    df = df.rename(columns={'precip_mm':'P','ash_mm_mean':'D'})
    df['P'] /= 1000.0
    df['D'] /= 1000.0

    # 3) initialize models
    nlrm = CatchmentNLRM(catch_df)
    dc   = build_network(edges_gdf)
    catch_df['H']      = 0.0
    catch_df['M_wash'] = 0.0

    discharge_ts: List[float] = []
    network_ts:   List[pd.DataFrame] = []

    # 4) loop through time
    for t, row in tqdm(df.iterrows(), total=len(df), desc="Hydro progress"):
        P_m, D_m = row['P'], row['D']
        catch_df['H'] += D_m

        # 4a) hydrology
        discharges = nlrm.update(P_m, evapotranspiration, catch_df['H'].to_dict())

        # 4b) ash wash on hillslopes
        comps = nlrm.get_flow_components()
        catch_df['M_wash'] = 0.0
        for cid, comp in comps.items():
            if comp['surface_flow'] > 0 and comp['total_outflow'] > 0:
                Ci = comp['surface_flow'] / comp['total_outflow']
                M_avail = rho_ash * catch_df.at[cid,'A_sub'] * catch_df.at[cid,'H']
                M_w = Ci * M_avail
                catch_df.at[cid,'M_wash'] = M_w
                catch_df.at[cid,'H'] *= (1 - Ci)

        # 4c) inject into reaches
        m = edges_gdf\
            .merge(catch_df[['M_wash']], left_on='catchment_id', right_index=True, how='left')\
            .fillna({'M_wash':0.0})
        A_sum = edges_gdf.groupby('catchment_id')['A_sub'].sum().reset_index(name='A_sum')
        m = m.merge(A_sum, on='catchment_id', how='left')
        m['M_edge'] = m['M_wash'] * (m['A_sub'] / m['A_sum'])

        for _, er in m.iterrows():
            rid = int(er['reach_id'])
            rc  = dc.reaches[rid]
            cid = er['catchment_id']
            Qc  = discharges.get(cid, 0.0)
            if cid in nlrm.shared_catchments:
                frac = er['A_sub']/er['A_sum']
                rc.discharge = Qc * frac
            else:
                rc.discharge = Qc
            rc.add_sediment_mass('ash', er['M_edge'] * 1000.0)

        # 4d) propagate and record
        flux: Dict[int, float] = {}
        for rid in dc.topo_order:
            inflow = sum(flux.get(u,0.0) for u in dc._upstream_ids(rid))
            flux[rid] = dc.reaches[rid].discharge + inflow

        # record mean outlet Q
        outlet = next(r for r in dc.topo_order if dc.reaches[r].downstream_id is None)
        Q_out = flux[outlet] / 86400.0
        discharge_ts.append(Q_out)

        # snapshot network state
        snap = edges_gdf.set_index('reach_id').loc[dc.topo_order].reset_index()
        snap['discharge']   = [dc.reaches[r].discharge for r in dc.topo_order]
        snap['ash_storage'] = [dc.reaches[r].storage['ash'] for r in dc.topo_order]
        snap['time_step']   = t
        network_ts.append(snap)

        # step sediment routing
        dc.step(dt=1.0)

    return {
        'discharge_ts': discharge_ts,
        'network_ts':   network_ts,
        'nlrm':         nlrm,
    }