# engines/visualize.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize, LogNorm
from tqdm import tqdm

def timeseries(precip_df, ash_df, results, outfile):
    """
    Creates a four-panel plot showing rainfall, discharge, hillslope ash,
    and cumulative ash input to the river network.
    """
    # --- Process all timeseries from the results dictionary ---
    df_plot = precip_df.copy()

    # Process water discharge
    q_ts = results.get('discharge_ts')
    if q_ts is not None and not q_ts.empty:
        daily_q = q_ts.set_index('datetime')['discharge_m3s'].resample('D').mean().reset_index()
        daily_q.rename(columns={'datetime': 'date', 'discharge_m3s': 'Q'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_q, on='date', how='left')

    # Process ash discharge
    aq_ts = results.get('ash_discharge_ts')
    if aq_ts is not None and not aq_ts.empty:
        daily_aq = aq_ts.set_index('datetime')['ash_flux_kg_s'].resample('D').mean().reset_index()
        daily_aq.rename(columns={'datetime': 'date', 'ash_flux_kg_s': 'Q_ash'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_aq, on='date', how='left')

    # Process hillslope ash depth
    ha_ts = results.get('hillslope_ash_depth_ts')
    if ha_ts is not None and not ha_ts.empty:
        daily_ha = ha_ts.set_index('datetime')['mean_ash_depth_mm'].resample('D').mean().reset_index()
        daily_ha.rename(columns={'datetime': 'date'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_ha, on='date', how='left')

    # Process cumulative ash wash
    cw_ts = results.get('cumulative_ash_wash_ts')
    if cw_ts is not None and not cw_ts.empty:
        daily_cw = cw_ts.set_index('datetime')['cumulative_ash_wash_kg'].resample('D').last().reset_index() # Use last value of the day
        daily_cw.rename(columns={'datetime': 'date'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_cw, on='date', how='left')

    # Process total ash storage in the river network
    snapshots = results.get("network_snapshots") or results.get("network_ts")
    storage_over_time = []
    if snapshots:
        is_nlrm = 'time_step' in snapshots[0].columns and 'datetime' not in snapshots[0].columns
        start_date = df_plot['date'].min()
        for snap in snapshots:
            total_storage = snap['ash_storage'].sum()
            current_date = (start_date + pd.Timedelta(days=snap['time_step'].iloc[0])) if is_nlrm else snap['datetime'].iloc[0].normalize()
            storage_over_time.append({'date': current_date, 'total_storage': total_storage})
        if storage_over_time:
            storage_df = pd.DataFrame(storage_over_time).groupby('date')['total_storage'].mean().reset_index()
            df_plot = pd.merge(df_plot, storage_df, on='date', how='left')

    # Fill any missing values
    df_plot = df_plot.fillna(method='ffill').fillna(0)

    # --- Plotting ---
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, figsize=(12, 10))
    fig.suptitle("Hydrology and Ash Transport Diagnostics", fontsize=16)

    # Panel 1: Rainfall
    ax1.bar(df_plot['date'], df_plot['precip_mm'], label='Rainfall')
    ax1.invert_yaxis()
    ax1.set_ylabel('Rain (mm/day)')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Discharge (Water and Ash)
    ax2.plot(df_plot['date'], df_plot.get('Q', 0), color='b', label='Water Discharge')
    ax2.set_ylabel('Water Q (m³/s)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)

    ax2b = ax2.twinx()
    ax2b.plot(df_plot['date'], df_plot.get('Q_ash', 0), color='saddlebrown', linestyle='--', label='Ash Discharge')
    ax2b.set_ylabel('Ash Q (kg/s)', color='saddlebrown')
    ax2b.tick_params(axis='y', labelcolor='saddlebrown')
    ax2b.set_ylim(bottom=0)

    # Panel 3: Ash on Hillslope vs. Ash in River
    ax3.plot(df_plot['date'], df_plot.get('mean_ash_depth_mm', 0), color='green', label='Mean Ash on Hillslope')
    ax3.set_ylabel('Hillslope Ash (mm)')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=0)

    ax3b = ax3.twinx()
    ax3b.plot(df_plot['date'], df_plot.get('total_storage', 0), color='purple', linestyle='--', label='Total Ash in River')
    ax3b.set_ylabel('River Storage (kg)', color='purple')
    ax3b.tick_params(axis='y', labelcolor='purple')
    ax3b.set_ylim(bottom=0)
    fig.legend(loc="upper right", bbox_to_anchor=(1,1), bbox_transform=ax3.transAxes)

    # Panel 4: Cumulative Ash Washed into River
    ax4.plot(df_plot['date'], df_plot.get('cumulative_ash_wash_kg', 0), color='brown', label='Cumulative Ash Input')
    ax4.set_ylabel('Cumulative Input (kg)')
    ax4.set_xlabel('Date')
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)

def plot_stock_flow(precip_df, results, outfile, tag: str = None):
    """
    Creates a publication-quality, three-panel plot showing the relationship
    between hydrologic flow (discharge) and sediment stock (storage).

    Panel A: Driving rainfall
    Panel B: Water and ash discharge (Flows)
    Panel C: Ash storage in the network (Stock)
    """
    # --- 1. Process and merge all required timeseries data ---
    df_plot = precip_df.copy()

    # Process water discharge (Q)
    q_ts = results.get('discharge_ts')
    if q_ts is not None and not q_ts.empty:
        daily_q = q_ts.set_index('datetime')['discharge_m3s'].resample('D').mean().reset_index()
        daily_q.rename(columns={'datetime': 'date', 'discharge_m3s': 'Q_water'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_q, on='date', how='left')

    # Process ash discharge (Q_ash)
    aq_ts = results.get('ash_discharge_ts')
    if aq_ts is not None and not aq_ts.empty:
        daily_aq = aq_ts.set_index('datetime')['ash_flux_kg_s'].resample('D').mean().reset_index()
        daily_aq.rename(columns={'datetime': 'date', 'ash_flux_kg_s': 'Q_ash'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_aq, on='date', how='left')

    # Process cumulative ash washed from hillslopes
    cw_ts = results.get('cumulative_ash_wash_ts')
    if cw_ts is not None and not cw_ts.empty:
        daily_cw = cw_ts.set_index('datetime')['cumulative_ash_wash_kg'].resample('D').last().reset_index()
        daily_cw.rename(columns={'datetime': 'date'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_cw, on='date', how='left')

    # Process total ash stored in the river network
    snapshots = results.get("network_snapshots") or results.get("network_ts")
    storage_over_time = []
    if snapshots:
        is_nlrm = 'time_step' in snapshots[0].columns and 'datetime' not in snapshots[0].columns
        start_date = df_plot['date'].min()
        for snap in snapshots:
            total_storage = snap['ash_storage'].sum()
            current_date = (start_date + pd.Timedelta(days=snap['time_step'].iloc[0])) if is_nlrm else snap['datetime'].iloc[0].normalize()
            storage_over_time.append({'date': current_date, 'total_storage_kg': total_storage})
        if storage_over_time:
            storage_df = pd.DataFrame(storage_over_time).groupby('date')['total_storage_kg'].mean().reset_index()
            df_plot = pd.merge(df_plot, storage_df, on='date', how='left')

    # Fill any missing values for plotting
    df_plot = df_plot.fillna(method='ffill').fillna(0)

    # --- 2. Create the 3-panel plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(10, 8),
        gridspec_kw={'height_ratios': [1, 2, 2]} # Give more space to lower panels
    )
    
    # NEW: Add the simulation tag as a main title
    if tag:
        fig.suptitle(f"Stock and Flow Relationship\nSimulation: {tag}", fontsize=14)
        fig.tight_layout(pad=4.0, rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout(pad=4.0)

    # --- Panel (a): Rainfall ---
    ax1.bar(df_plot['date'], df_plot['precip_mm'], color='cornflowerblue', width=1.0)
    ax1.set_ylabel('Rainfall\n(mm/day)')
    ax1.invert_yaxis()
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.text(0.02, 0.95, '(a)', transform=ax1.transAxes, fontsize=12, va='top', ha='left', weight='bold')


    # --- Panel (b): Flows (Water and Ash Discharge) ---
    ax2.plot(df_plot['date'], df_plot.get('Q_water', 0), color='royalblue', label='Water Discharge ($Q$)')
    ax2.set_ylabel('Water Discharge, $Q$\n(m$^3$/s)', color='royalblue')
    ax2.tick_params(axis='y', labelcolor='royalblue')
    ax2.set_ylim(bottom=0)
    ax2.grid(True, linestyle='--', alpha=0.6)

    ax2b = ax2.twinx()
    ax2b.plot(df_plot['date'], df_plot.get('Q_ash', 0), color='saddlebrown', linestyle='--', label='Ash Discharge ($Q_a$)')
    ax2b.set_ylabel('Ash Discharge, $Q_a$\n(kg/s)', color='saddlebrown')
    ax2b.tick_params(axis='y', labelcolor='saddlebrown')
    ax2b.set_ylim(bottom=0)
    ax2.text(0.02, 0.95, '(b)', transform=ax2.transAxes, fontsize=12, va='top', ha='left', weight='bold')

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2b.legend(lines + lines2, labels + labels2, loc='upper right')


    # --- Panel (c): Stock (Ash Storage) ---
    ax3.plot(df_plot['date'], df_plot.get('total_storage_kg', 0), color='purple', label='Ash Storage in Network')
    ax3.set_ylabel('Ash Storage in Network\n(kg)', color='purple')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3.set_ylim(bottom=0)
    ax3.set_xlabel('Date')

    ax3b = ax3.twinx()
    ax3b.plot(df_plot['date'], df_plot.get('cumulative_ash_wash_kg', 0), color='darkgreen', linestyle=':', label='Cumulative Hillslope Input')
    ax3b.set_ylabel('Cumulative Hillslope Input\n(kg)', color='darkgreen')
    ax3b.tick_params(axis='y', labelcolor='darkgreen')
    ax3b.set_ylim(bottom=0)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=12, va='top', ha='left', weight='bold')

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3b.legend(lines + lines2, labels + labels2, loc='upper right')


    # --- Save the figure ---
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)

def network_animation(results, outfile):
    """
    Animate network discharge (water flow) and ash flow side-by-side over time.
    """
    if 'static_network_gdf' in results:
        snapshots = results.get("network_snapshots")
        static_gdf = results.get("static_network_gdf")
    else:
        snapshots = results.get("network_ts")
        static_gdf = None

    if not snapshots:
        print("No network timeseries data found for animation.")
        return

    print("Calculating animation normalization ranges...")
    max_q, max_ash_flow = 0, 0
    min_q_nonzero, min_ash_flow_nonzero = float('inf'), float('inf')

    for df in tqdm(snapshots, desc="Analyzing frames"):
        # --- For Water Discharge ---
        q_series = df['discharge_m3s']
        if not q_series.empty:
            max_q = max(max_q, q_series.max())
            nonzero_q = q_series[q_series > 0]
            if not nonzero_q.empty:
                min_q_nonzero = min(min_q_nonzero, nonzero_q.min())

        # --- For Ash Flow ---
        if 'ash_flow_kgs' in df.columns:
            a_series = df['ash_flow_kgs']
            if not a_series.empty:
                max_ash_flow = max(max_ash_flow, a_series.max())
                nonzero_a = a_series[a_series > 0]
                if not nonzero_a.empty:
                    min_ash_flow_nonzero = min(min_ash_flow_nonzero, nonzero_a.min())

    if max_q == 0: max_q = 1.0
    if min_q_nonzero == float('inf'): min_q_nonzero = 1e-6
    
    if max_ash_flow == 0: max_ash_flow = 1.0
    if min_ash_flow_nonzero == float('inf'): min_ash_flow_nonzero = 1e-6

    norm_q = LogNorm(vmin=min_q_nonzero, vmax=max_q, clip=True)
    norm_ash_flow = LogNorm(vmin=min_ash_flow_nonzero, vmax=max_ash_flow, clip=True)

    cmap_q = plt.cm.viridis
    cmap_ash_flow = plt.cm.plasma

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
    fig.suptitle("Network Animation", fontsize=16)
    ax1.set_title("Water Discharge (m³/s)")
    ax2.set_title("Ash Flow (kg/s)") # UPDATED

    for ax in (ax1, ax2):
        ax.set_axis_off()

    sm_q = plt.cm.ScalarMappable(cmap=cmap_q, norm=norm_q)
    cbar1 = fig.colorbar(sm_q, ax=ax1, orientation='horizontal', pad=0.01)
    cbar1.set_label('Water Discharge (m³/s)')
    sm_a = plt.cm.ScalarMappable(cmap=cmap_ash_flow, norm=norm_ash_flow)
    cbar2 = fig.colorbar(sm_a, ax=ax2, orientation='horizontal', pad=0.01)
    cbar2.set_label('Ash Flow (kg/s)') # UPDATED
    
    def update(frame_index):
        snapshot_df = snapshots[frame_index]
        time_step = snapshot_df['time_step'].iloc[0]

        if static_gdf is not None:
            g = static_gdf.merge(snapshot_df, on='reach_id', how='left').fillna(0)
        else:
            g = snapshot_df

        ax1.clear(); ax2.clear()
        ax1.set_title(f"Water Discharge (Step {time_step})")
        ax2.set_title(f"Ash Flow (Step {time_step})") # UPDATED
        
        g.plot(ax=ax1, column='discharge_m3s', cmap=cmap_q, norm=norm_q, linewidth=2)
        g.plot(ax=ax2, column='ash_flow_kgs', cmap=cmap_ash_flow, norm=norm_ash_flow, linewidth=2) # UPDATED
        
        for ax in (ax1, ax2):
            ax.set_axis_off()

    ani = FuncAnimation(fig, update, frames=len(snapshots), interval=200)
    
    def progress_callback(i, n):
        print(f'Saving animation frame {i + 1} of {n}', end='\r')
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    ani.save(outfile, dpi=180, progress_callback=progress_callback)
    plt.close(fig)
    print(f"\nAnimation saved to {outfile}")