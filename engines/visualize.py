# engines/visualize.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
from matplotlib.animation import FuncAnimation
from matplotlib.colors import Normalize, LogNorm
from tqdm import tqdm
import numpy as np
import seaborn as sns



def plot_stock_flow(precip_df, results, outfile, tag: str = None):
    """
    Creates a publication-quality, three-panel plot showing the relationship
    between hydrologic flow, sediment stock, and hillslope ash reserves.
    """
    # --- 1. Process all required timeseries data ---
    df_plot = precip_df.copy()

    # Process water discharge (Q)
    q_ts = results.get('discharge_ts')
    if q_ts is not None and not q_ts.empty:
        # Create daily summary and RENAME datetime column to 'date' for merging
        daily_q = q_ts.set_index('datetime')['discharge_m3s'].resample('D').mean().reset_index()
        daily_q.rename(columns={'datetime': 'date'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_q.rename(columns={'discharge_m3s': 'Q_water'}), on='date', how='left')

    # Process ash discharge (Q_ash) to get daily mean and cumulative total
    aq_ts = results.get('ash_discharge_ts')
    if aq_ts is not None and not aq_ts.empty:
        # Calculate daily mean flux for panel (b)
        daily_aq_flux = aq_ts.set_index('datetime')['ash_flux_kg_s'].resample('D').mean().reset_index()
        daily_aq_flux.rename(columns={'datetime': 'date'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_aq_flux.rename(columns={'ash_flux_kg_s': 'Q_ash'}), on='date', how='left')
        
        # Calculate cumulative total discharged from the outlet for panel (c)
        aq_ts['daily_ash_kg'] = aq_ts['ash_flux_kg_s'] * 86400 # convert kg/s to kg/day
        aq_ts['cumulative_ash_outlet_kg'] = aq_ts['daily_ash_kg'].cumsum()
        daily_cumulative_outlet = aq_ts.set_index('datetime')['cumulative_ash_outlet_kg'].resample('D').last().reset_index()
        daily_cumulative_outlet.rename(columns={'datetime': 'date'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_cumulative_outlet, on='date', how='left')


    # Process total ash stored IN THE RIVER NETWORK
    snapshots = results.get("network_snapshots") or results.get("network_ts")
    if snapshots:
        storage_over_time = []
        start_date = df_plot['date'].min()
        for snap in snapshots:
            total_storage = snap['ash_storage'].sum()
            # This part correctly uses 'date' already
            current_date = (start_date + pd.Timedelta(days=snap['time_step'].iloc[0])) if 'time_step' in snap.columns else snap['datetime'].iloc[0].normalize()
            storage_over_time.append({'date': current_date, 'network_storage_kg': total_storage})
        if storage_over_time:
            storage_df = pd.DataFrame(storage_over_time).groupby('date')['network_storage_kg'].mean().reset_index()
            df_plot = pd.merge(df_plot, storage_df, on='date', how='left')

    # Process total ash remaining ON THE HILLSLOPES
    ha_ts = results.get('hillslope_ash_depth_ts')
    scscn_model = results.get('scscn_model')
    if ha_ts is not None and scscn_model is not None:
        total_catchment_area = sum(c.area_m2 for c in scscn_model.catchments.values())
        ha_ts['total_hillslope_ash_kg'] = (ha_ts['mean_ash_depth_mm'] / 1000) * total_catchment_area * 1000 # rho_ash
        daily_hillslope_ash = ha_ts.set_index('datetime')['total_hillslope_ash_kg'].resample('D').mean().reset_index()
        daily_hillslope_ash.rename(columns={'datetime': 'date'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_hillslope_ash, on='date', how='left')


    # Fill any missing values for plotting
    df_plot = df_plot.ffill().fillna(0)

    # --- 2. Create the 3-panel plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, sharex=True, figsize=(12, 9),
        gridspec_kw={'height_ratios': [1, 2, 2]}
    )
    if tag:
        fig.suptitle(f"Stock and Flow Relationship\nSimulation: {tag}", fontsize=14)
        fig.tight_layout(pad=3.0, rect=[0, 0, 1, 0.95])
    else:
        fig.tight_layout(pad=3.0)

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
    ax2b.legend(lines + lines2, labels + labels2, loc='center left')

    # --- Panel (c): Ash Stocks (Hillslope vs. Network vs. Discharged) ---
    # ax3.plot(df_plot['date'], df_plot.get('network_storage_kg', 0), color='purple', label='Ash Storage in Network')
    ax3.plot(df_plot['date'], df_plot.get('cumulative_ash_outlet_kg', 0), color='saddlebrown', linestyle=':', label='Cumulative Ash Discharged from Outlet')
    ax3.set_ylabel('Ash Mass (kg)')
    ax3.tick_params(axis='y')
    ax3.set_ylim(bottom=0)
    ax3.set_xlabel('Date')
    
    ax3b = ax3.twinx()
    ax3b.fill_between(df_plot['date'], df_plot.get('total_hillslope_ash_kg', 0), color='darkgreen', alpha=0.2, label='Total Ash on Hillslopes')
    ax3b.set_ylabel('Total Hillslope Ash (kg)', color='darkgreen')
    ax3b.tick_params(axis='y', labelcolor='darkgreen')
    ax3b.set_ylim(bottom=0)
    
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.text(0.02, 0.95, '(c)', transform=ax3.transAxes, fontsize=12, va='top', ha='left', weight='bold')

    lines, labels = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3b.get_legend_handles_labels()
    ax3b.legend(lines + lines2, labels + labels2, loc='center left')

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


def plot_precip_discharge_relationship(
    precip_df,
    results,
    outfile,
    ash_discharge_threshold_kgs=0.01,
    precip_threshold_mm=1.0,
    tag=None
):
    """
    Creates an enhanced scatter plot with robust regression lines to show the
    relationship between precipitation and discharge, colored by ash condition.
    """
    # --- 1. Process and merge all required timeseries data ---
    df_plot = precip_df.copy()

    q_ts = results.get('discharge_ts')
    if q_ts is not None and not q_ts.empty:
        daily_q = q_ts.set_index('datetime')['discharge_m3s'].resample('D').max().reset_index()
        daily_q.rename(columns={'datetime': 'date', 'discharge_m3s': 'Q_water'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_q, on='date', how='left')

    aq_ts = results.get('ash_discharge_ts')
    if aq_ts is not None and not aq_ts.empty:
        daily_aq = aq_ts.set_index('datetime')['ash_flux_kg_s'].resample('D').mean().reset_index()
        daily_aq.rename(columns={'datetime': 'date', 'ash_flux_kg_s': 'Q_ash'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_aq, on='date', how='left')

    df_plot = df_plot.fillna(0)

    # --- 2. Filter data for meaningful events ---
    df_plot = df_plot[df_plot['precip_mm'] > precip_threshold_mm]

    # --- 3. Categorize each day based on ash discharge ---
    df_plot['Ash Condition'] = np.where(
        df_plot['Q_ash'] > ash_discharge_threshold_kgs,
        f'With Ash (> {ash_discharge_threshold_kgs:.2f} kg/s)',
        'No Ash'
    )

    # --- 4. Create the regression plot using seaborn ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(11, 8))


    # We will add the `robust=True` parameter to the regplot call for the "With Ash" data.
    # This tells seaborn to use a regression model that is not sensitive to outliers.

    sns.regplot(
        x='precip_mm', y='Q_water',
        data=df_plot[df_plot['Ash Condition'] == 'No Ash'],
        ax=ax,
        label='No Ash',
        scatter_kws={'alpha': 0.6, 's': 50, 'edgecolor': 'k'},
        line_kws={'linewidth': 2.5}
    )
    sns.regplot(
        x='precip_mm', y='Q_water',
        data=df_plot[df_plot['Ash Condition'] != 'No Ash'],
        ax=ax,
        label=f'With Ash (> {ash_discharge_threshold_kgs:.2f} kg/s)',
        scatter_kws={'alpha': 0.8, 's': 70, 'edgecolor': 'k'},
        line_kws={'linewidth': 2.5, 'linestyle': '--'},
        marker='D',
        robust=True  # <-- ADD THIS LINE TO FIX THE TREND
    )

    ax.set_xlabel('Daily Precipitation (mm)', fontsize=12)
    ax.set_ylabel('Peak Daily Discharge (m³/s)', fontsize=12)
    title = "Ash Impact on Precipitation-Discharge Relationship"
    if tag:
        title += f"\n(Simulation: {tag})"
    ax.set_title(title, fontsize=14, weight='bold')
    ax.legend()
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    # --- 5. Save the figure ---
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_event_hydrographs(
    results,
    precip_df,
    outfile,
    ash_event_day: int,
    precip_min_mm: float = 5.0,
):
    """
    Compares the hydrograph response of two separate storm events,
    one pre-ash and one post-ash, found automatically.
    """
    hourly_df = results.get('hourly_data')
    if hourly_df is None or 'precip_mm' not in hourly_df.columns:
        print("Hourly precipitation data not found in results. Cannot create event plot.")
        return
        
    q_ts = results.get('discharge_ts')
    if q_ts is None:
        print("Discharge timeseries not found. Cannot create event plot.")
        return

    # --- Automatically find pre- and post-ash storm events ---
    start_date = hourly_df['datetime'].min()
    ash_event_date = start_date + pd.Timedelta(days=ash_event_day)
    
    daily_precip = hourly_df.set_index('datetime')['precip_mm'].resample('D').sum()

    # Find the last significant storm before the ash event
    pre_ash_storms = daily_precip[(daily_precip.index < ash_event_date) & (daily_precip > precip_min_mm)]
    if pre_ash_storms.empty:
        print(f"Could not find a suitable pre-ash storm event (>{precip_min_mm}mm) to plot.")
        return
    pre_ash_event_date_str = pre_ash_storms.index[-1].strftime('%Y-%m-%d')

    # Find the first significant storm after the ash event
    post_ash_storms = daily_precip[(daily_precip.index > ash_event_date) & (daily_precip > precip_min_mm)]
    if post_ash_storms.empty:
        print(f"Could not find a suitable post-ash storm event (>{precip_min_mm}mm) to plot.")
        return
    post_ash_event_date_str = post_ash_storms.index[0].strftime('%Y-%m-%d')
    
    print(f"Comparing pre-ash storm on {pre_ash_event_date_str} with post-ash storm on {post_ash_event_date_str}")

    # --- Plotting logic (mostly unchanged) ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(10, 8),
        gridspec_kw={'height_ratios': [1, 2]}
    )
    fig.suptitle("Hydrograph Response Comparison: Pre-Ash vs. Post-Ash", fontsize=16)

    # --- Process and plot each event ---
    for event_date_str, color, style, label in [
        (pre_ash_event_date_str, 'royalblue', '-', 'Pre-Ash Event'),
        (post_ash_event_date_str, 'orangered', '--', 'Post-Ash Event')
    ]:
        event_date = pd.to_datetime(event_date_str)
        
        # Get data for a 48-hour window around the event
        start_time = event_date
        end_time = start_time + pd.Timedelta(days=2)
        
        # FIX: Add .copy() to prevent SettingWithCopyWarning
        precip_event = hourly_df[(hourly_df['datetime'] >= start_time) & (hourly_df['datetime'] < end_time)].copy()
        q_event = q_ts[(q_ts['datetime'] >= start_time) & (q_ts['datetime'] < end_time)].copy()
        
        # Calculate hours from start for plotting
        precip_event['hours'] = (precip_event['datetime'] - start_time).dt.total_seconds() / 3600
        q_event['hours'] = (q_event['datetime'] - start_time).dt.total_seconds() / 3600
        
        # Panel 1: Rainfall
        ax1.plot(precip_event['hours'], precip_event['precip_mm'], color=color, linestyle=style, label=f"{label} Rainfall")
        
        # Panel 2: Discharge
        ax2.plot(q_event['hours'], q_event['discharge_m3s'], color=color, linestyle=style, label=f"{label} Discharge", linewidth=2.5)

    # --- Formatting ---
    ax1.set_ylabel("Rainfall Rate\n(mm/hr)")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    ax2.set_ylabel("Outlet Discharge (m³/s)")
    ax2.set_xlabel("Hours from Storm Start")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()
    ax2.set_ylim(bottom=0)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Event comparison plot saved to {outfile}")



def plot_response_spyder(results, precip_df, outfile):
    """
    Creates a spyder (radar) chart by splitting a single simulation run
    into 'Pre-Ash' and 'Post-Ash' periods to compare discharge response.

    Args:
        results (dict): The full results from the single simulation with an eruption.
        precip_df (pd.DataFrame): The precipitation dataframe used for the simulation.
        outfile (Path): The file path to save the plot.
    """
    print("Generating spyder chart from single simulation run...")

    # --- 1. Data Preparation ---
    eruption_days = results.get('eruption_days')
    if not eruption_days:
        print("No eruption days found in results. Cannot create spyder chart.")
        return

    # Use the first eruption day to split the data
    first_eruption_day = eruption_days[0]

    q_ts = results.get('discharge_ts')
    if q_ts is None or q_ts.empty:
        print("Discharge timeseries not found in results.")
        return

    # Get peak daily discharge
    peak_daily_q = q_ts.set_index('datetime')['discharge_m3s'].resample('D').max().reset_index()
    peak_daily_q.rename(columns={'datetime': 'date', 'discharge_m3s': 'peak_q'}, inplace=True)

    # Merge with daily precipitation
    data = pd.merge(precip_df, peak_daily_q, on='date', how='left').fillna(0)
    
    # --- 2. Split data into Pre-Ash and Post-Ash periods ---
    start_date = data['date'].min()
    eruption_date = start_date + pd.Timedelta(days=first_eruption_day)

    pre_ash_data = data[data['date'] < eruption_date]
    post_ash_data = data[data['date'] >= eruption_date]

    # --- 3. Process each period separately ---
    def _calculate_response(df, labels):
        """Helper function to calculate response for a given data period."""
        if df.empty:
            return pd.Series(index=labels, data=0.0)
            
        bins = [0, 5, 10, 15, 20, np.inf]
        df['precip_category'] = pd.cut(df['precip_mm'], bins=bins, labels=labels, right=False)
        
        response_data = df[(df['precip_mm'] > 0) & (df['peak_q'] > 0)]
        if response_data.empty:
            return pd.Series(index=labels, data=0.0)

        avg_peak_q = response_data.groupby('precip_category', observed=False)['peak_q'].mean()
        return avg_peak_q.reindex(labels).fillna(0)

    category_labels = ["Light Rain (0-5mm)", "Moderate (5-10mm)", "Heavy (10-15mm)", "Very Heavy (15-20mm)", "Extreme (>20mm)"]
    response_post_ash = _calculate_response(post_ash_data.copy(), category_labels)
    response_pre_ash = _calculate_response(pre_ash_data.copy(), category_labels)

    # --- 4. Plot the Spyder Chart ---
    plot_data = pd.DataFrame({'Post-Ash': response_post_ash, 'Pre-Ash (baseline)': response_pre_ash})
    categories = plot_data.index.tolist()
    num_vars = len(categories)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1] # Close the plot

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    # Plot "Pre-Ash" data (baseline)
    values_pre_ash = plot_data['Pre-Ash (baseline)'].tolist()
    values_pre_ash += values_pre_ash[:1]
    ax.plot(angles, values_pre_ash, color='royalblue', linewidth=1, linestyle='solid', label='Pre-Ash (baseline)')
    ax.fill(angles, values_pre_ash, color='royalblue', alpha=0.25)

    # Plot "Post-Ash" data
    values_post_ash = plot_data['Post-Ash'].tolist()
    values_post_ash += values_post_ash[:1]
    ax.plot(angles, values_post_ash, color='orangered', linewidth=1, linestyle='solid', label='Post-Ash')
    ax.fill(angles, values_post_ash, color='orangered', alpha=0.25)

    # Formatting
    ax.set_yticklabels([])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_rlabel_position(180 / num_vars)
    
    max_val = plot_data.max().max()
    if max_val > 0:
        for i in np.arange(0, max_val * 1.1, max_val/4):
             ax.text(0, i, f'{i:.0f} m³/s', ha='center', va='center', fontsize=8, color='gray')

    plt.title('Discharge Response to Rainfall Intensity', size=16, color='black', y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Spyder chart saved to {outfile}")


def plot_runoff_coefficient(results, precip_df, outfile):
    """
    Plots the time series of the daily runoff coefficient (Q/P) against
    rainfall and a dynamic representation of hillslope ash depth.
    """
    # --- 1. Process and merge data ---
    df_plot = precip_df.copy()

    # Process water discharge to get daily total volume
    q_ts = results.get('discharge_ts')
    if q_ts is not None and not q_ts.empty:
        q_ts['Q_vol_m3'] = q_ts['discharge_m3s'] * 3600 # m3 per hour
        daily_q = q_ts.set_index('datetime')['Q_vol_m3'].resample('D').sum().reset_index()
        daily_q.rename(columns={'datetime': 'date'}, inplace=True)
        df_plot = pd.merge(df_plot, daily_q, on='date', how='left')

    # Get daily MIN and MAX ash depth to show washout
    ha_ts = results.get('hillslope_ash_depth_ts')
    if ha_ts is not None and not ha_ts.empty:
        # Resample to get the maximum (start of day) and minimum (end of day) ash depth
        # FIX: Add .reset_index() to create a 'datetime' column for merging
        daily_ha_max = ha_ts.set_index('datetime')['mean_ash_depth_mm'].resample('D').max().reset_index().rename(columns={'datetime': 'date', 'mean_ash_depth_mm': 'ash_depth_max_mm'})
        daily_ha_min = ha_ts.set_index('datetime')['mean_ash_depth_mm'].resample('D').min().reset_index().rename(columns={'datetime': 'date', 'mean_ash_depth_mm': 'ash_depth_min_mm'})
        
        # Merge both into the main plot dataframe
        df_plot = pd.merge(df_plot, daily_ha_max, on='date', how='left')
        df_plot = pd.merge(df_plot, daily_ha_min, on='date', how='left')

    df_plot = df_plot.fillna(0)
    
    # --- 2. Calculate Runoff Coefficient (C) ---
    df_plot['runoff_coeff'] = 0.0
    rainy_days_mask = df_plot['precip_mm'] > 1.0 
    
    # Use a placeholder for basin area for this approximate calculation
    # A more accurate C would require the total basin area
    basin_area_proxy = 10000 
    # Pandas recommends using .loc for this type of assignment to avoid SettingWithCopyWarning
    df_plot.loc[rainy_days_mask, 'runoff_coeff'] = (df_plot.loc[rainy_days_mask, 'Q_vol_m3'] / 
                                                   (df_plot.loc[rainy_days_mask, 'precip_mm'] / 1000 * basin_area_proxy))
    df_plot['runoff_coeff'] = df_plot['runoff_coeff'].clip(0, 1.5)

    # --- 3. Create the 3-panel plot ---
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 8),
                                      gridspec_kw={'height_ratios': [1, 2, 2]})
    fig.suptitle("Runoff Coefficient Response to Ashfall", fontsize=16)

    # Panel 1: Rainfall
    ax1.bar(df_plot['date'], df_plot['precip_mm'], color='cornflowerblue')
    ax1.set_ylabel("Rainfall\n(mm/day)")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot ash depth as a shaded min/max region to show washout
    ax2.fill_between(
        df_plot['date'],
        df_plot.get('ash_depth_min_mm', 0),
        df_plot.get('ash_depth_max_mm', 0),
        color='saddlebrown',
        alpha=0.4,
        label='Hillslope Ash Range (Min/Max)'
    )
    ax2.plot(df_plot['date'], df_plot.get('ash_depth_max_mm', 0), color='saddlebrown', alpha=0.7)
    ax2.set_ylabel("Mean Ash Depth\n(mm)")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_ylim(bottom=0)
    ax2.legend()

    # Plot runoff coefficient as a scatter plot for rainy days only
    rainy_df = df_plot[rainy_days_mask]
    ax3.scatter(
        rainy_df['date'],
        rainy_df['runoff_coeff'],
        color='crimson',
        label='Runoff Coefficient (on rainy days)',
        s=25,
        alpha=0.7
    )
    ax3.set_ylabel("Runoff Coeff. (C)")
    ax3.set_xlabel("Date")
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.set_ylim(bottom=0, top=1.6)
    ax3.legend()
    
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Runoff coefficient plot saved to {outfile}")



    # Add this new function to the end of engines/visualize.py


def plot_hillslope_ash_decay(results, precip_df, outfile):
    """
    Creates a dedicated plot to visualize the decay of ash on hillslopes over time,
    using a log scale to highlight the rate of change.

    Args:
        results (dict): The results from a simulation run.
        precip_df (pd.DataFrame): The precipitation dataframe.
        outfile (Path): The file path to save the plot.
    """
    print("Generating hillslope ash decay plot...")

    ha_ts = results.get('hillslope_ash_depth_ts')
    if ha_ts is None or ha_ts.empty:
        print("Hillslope ash timeseries data not found. Cannot create decay plot.")
        return

    # Merge with daily precipitation for plotting
    plot_df = pd.merge(precip_df, ha_ts, left_on='date', right_on='datetime', how='left')

    # --- Create the Plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(12, 8),
        gridspec_kw={'height_ratios': [1, 2]}
    )
    fig.suptitle("Hillslope Ash Depth Decay Over Time", fontsize=16)

    # Panel 1: Rainfall
    ax1.bar(plot_df['date'], plot_df['precip_mm'], color='cornflowerblue')
    ax1.set_ylabel("Rainfall\n(mm/day)")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Ash Depth Decay with Log Scale
    ax2.plot(plot_df['date'], plot_df['mean_ash_depth_mm'], color='saddlebrown', label='Hillslope Ash Depth')
    
    # Use a logarithmic scale on the y-axis to better see decay
    ax2.set_yscale('log')
    
    ax2.set_ylabel("Mean Ash Depth (mm)\n(Log Scale)")
    ax2.set_xlabel("Date")
    ax2.grid(True, which='both', linestyle='--', alpha=0.5) # Grid on major and minor ticks
    ax2.legend()
    
    # Set a minimum threshold for the y-axis to avoid issues with zero values on a log scale
    # Find the minimum non-zero ash depth to set a reasonable bottom limit
    min_val = plot_df[plot_df['mean_ash_depth_mm'] > 0]['mean_ash_depth_mm'].min()
    if pd.notna(min_val):
        ax2.set_ylim(bottom=min_val * 0.9)


    # --- Save the figure ---
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300)
    plt.close(fig)
    print(f"Hillslope ash decay plot saved to {outfile}")




    # Add this new function to the end of engines/visualize.py


# In engines/visualize.py, replace the existing function with this one.

def plot_hydrograph_ensemble(results, precip_df, outfile, precip_threshold_mm=10.0):
    """
    Plots an ensemble of hydrograph responses to significant rainfall events,
    showing the mean and maximum response before and after an eruption.

    Args:
        results (dict): The results from a simulation run.
        precip_df (pd.DataFrame): The precipitation dataframe.
        outfile (Path): The file path to save the plot.
        precip_threshold_mm (float): The daily rainfall total to define a 'significant' event.
    """
    print(f"Generating hydrograph ensemble plot for events > {precip_threshold_mm}mm/day...")

    # --- 1. Get required data ---
    eruption_days = results.get('eruption_days')
    if not eruption_days:
        print("No eruption days found in results. Cannot create ensemble plot.")
        return

    q_ts = results.get('discharge_ts')
    if q_ts is None or q_ts.empty:
        print("Discharge timeseries not found in results.")
        return
    q_ts = q_ts.set_index('datetime')

    # --- 2. Find all significant storm events ---
    significant_events = precip_df[precip_df['precip_mm'] > precip_threshold_mm]['date']
    
    # --- 3. Separate events into Pre-Ash and Post-Ash groups ---
    eruption_date = pd.to_datetime(precip_df['date'].min()) + pd.Timedelta(days=eruption_days[0])
    pre_ash_events = significant_events[significant_events < eruption_date]
    post_ash_events = significant_events[significant_events >= eruption_date]

    if pre_ash_events.empty or post_ash_events.empty:
        print("Could not find significant storm events in both pre- and post-ash periods.")
        return

    # --- 4. Process the ensembles ---
    def _create_ensemble_df(event_dates, q_ts):
        """Helper to extract and align hydrographs for a list of event dates."""
        ensemble_data = []
        for event_date in event_dates:
            # Define the 144-hour window centered on the middle of the event day
            start_time = event_date + pd.Timedelta(hours=12) - pd.Timedelta(hours=72)
            end_time = event_date + pd.Timedelta(hours=12) + pd.Timedelta(hours=72)
            
            hydrograph_slice = q_ts[(q_ts.index >= start_time) & (q_ts.index < end_time)]
            if hydrograph_slice.empty: continue
            
            # Create a relative time axis in hours from -72 to +72
            relative_hours = (hydrograph_slice.index - (event_date + pd.Timedelta(hours=12))).total_seconds() / 3600
            hydrograph_slice = hydrograph_slice.assign(relative_hours=relative_hours)
            ensemble_data.append(hydrograph_slice.set_index('relative_hours')['discharge_m3s'])

        if not ensemble_data: return None, None
        
        # Combine all events into a single dataframe, aligning by the relative hour index
        ensemble_df = pd.concat(ensemble_data, axis=1).interpolate(method='linear')
        
        # MODIFICATION: Calculate mean and max response
        mean_response = ensemble_df.mean(axis=1)
        max_response = ensemble_df.max(axis=1)
        
        return mean_response, max_response

    pre_mean, pre_max = _create_ensemble_df(pre_ash_events, q_ts)
    post_mean, post_max = _create_ensemble_df(post_ash_events, q_ts)

    # --- 5. Create the Plot ---
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot Pre-Ash Ensemble
    if pre_mean is not None:
        # MODIFICATION: Fill between mean and max
        ax.fill_between(pre_mean.index, pre_mean, pre_max, color='royalblue', alpha=0.2, label='Pre-Ash (Mean to Max Range)')
        ax.plot(pre_mean.index, pre_mean, color='royalblue', linestyle='-', linewidth=2.5, label='Pre-Ash (Mean Response)')

    # Plot Post-Ash Ensemble
    if post_mean is not None:
        # MODIFICATION: Fill between mean and max
        ax.fill_between(post_mean.index, post_mean, post_max, color='orangered', alpha=0.2, label='Post-Ash (Mean to Max Range)')
        ax.plot(post_mean.index, post_mean, color='orangered', linestyle='--', linewidth=2.5, label='Post-Ash (Mean Response)')

    # Formatting
    ax.axvline(0, color='black', linestyle=':', linewidth=1.5, label='Peak Rainfall Day')
    ax.set_title(f'Ensemble Hydrograph Response to Significant Rainfall (> {precip_threshold_mm}mm/day)', fontsize=16)
    ax.set_xlabel('Hours Relative to Peak Rainfall Day', fontsize=12)
    ax.set_ylabel('Discharge (m³/s)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend()
    ax.set_xlim(-72, 72)
    ax.set_ylim(bottom=0)

    # --- Save the figure ---
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Ensemble hydrograph plot saved to {outfile}")