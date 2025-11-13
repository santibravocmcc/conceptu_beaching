# =============================================================================
# INTERACTIVE BEACHING ANALYSIS DASHBOARD
# =============================================================================
# Streamlit application for testing thresholds and visualizing beaching data
# =============================================================================

import streamlit as st
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, BoundaryNorm
import folium
from folium import plugins
import pandas as pd
import os
import glob
from datetime import datetime
import io
import zipfile
import cmocean

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Beaching Analysis Dashboard",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Directory containing netCDF files
DATA_DIR = "beaching_data_nc"

# Histogram parameters
N_BINS = 20

# Month information
MONTH_INFO = {
    1: {"name": "January", "days": 31, "abbr": "JAN"},
    2: {"name": "February", "days": 28, "abbr": "FEB"},
    3: {"name": "March", "days": 31, "abbr": "MAR"},
    4: {"name": "April", "days": 30, "abbr": "APR"},
    5: {"name": "May", "days": 31, "abbr": "MAY"},
    6: {"name": "June", "days": 30, "abbr": "JUN"},
    7: {"name": "July", "days": 31, "abbr": "JUL"},
    8: {"name": "August", "days": 31, "abbr": "AUG"},
    9: {"name": "September", "days": 30, "abbr": "SEP"},
    10: {"name": "October", "days": 31, "abbr": "OCT"},
    11: {"name": "November", "days": 30, "abbr": "NOV"},
    12: {"name": "December", "days": 31, "abbr": "DEC"}
}

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

@st.cache_data
def load_available_months():
    """Find all available netCDF files."""
    nc_files = glob.glob(os.path.join(DATA_DIR, "beaching_*.nc"))
    
    available_months = {}
    for filepath in nc_files:
        filename = os.path.basename(filepath)
        # Extract month number from filename (beaching_01_january.nc)
        month_num = int(filename.split('_')[1])
        available_months[month_num] = filepath
    
    return available_months

@st.cache_data
def load_month_data(filepath):
    """Load data for a specific month."""
    ds = xr.open_dataset(filepath)
    return ds

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def calculate_threshold_statistics(total_particles_per_cell, thresholds):
    """Calculate statistics for each threshold."""
    total_cells = len(total_particles_per_cell)
    active_cells_baseline = np.sum(total_particles_per_cell > 0)
    
    stats = {}
    active_loss_stats = {}
    
    for threshold in thresholds:
        active_cells = np.sum(total_particles_per_cell >= threshold)
        inactive_cells = total_cells - active_cells
        
        stats[threshold] = {
            'active_cells': active_cells,
            'inactive_cells': inactive_cells,
            'percentage_kept': (active_cells / total_cells) * 100 if total_cells > 0 else 0,
            'percentage_lost': (inactive_cells / total_cells) * 100 if total_cells > 0 else 0
        }
        
        # Active cell loss statistics
        cells_kept = active_cells
        cells_lost_from_active = active_cells_baseline - cells_kept
        percentage_of_active_kept = (cells_kept / active_cells_baseline) * 100 if active_cells_baseline > 0 else 0
        percentage_of_active_lost = (cells_lost_from_active / active_cells_baseline) * 100 if active_cells_baseline > 0 else 0
        
        active_loss_stats[threshold] = {
            'cells_kept_from_active': cells_kept,
            'cells_lost_from_active': cells_lost_from_active,
            'percentage_of_active_kept': percentage_of_active_kept,
            'percentage_of_active_lost': percentage_of_active_lost
        }
    
    return stats, active_loss_stats, active_cells_baseline

def create_distribution_histogram(active_particles, thresholds, month_name, active_cells_baseline):
    """Create distribution histogram."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if len(active_particles) > 0:
        bins = np.logspace(0, np.log10(active_particles.max()), N_BINS)
    else:
        bins = [0, 1]
    
    counts, bin_edges = np.histogram(active_particles, bins=bins)
    counts_fraction = counts / active_cells_baseline if active_cells_baseline > 0 else counts
    
    ax.bar(bin_edges[:-1], counts_fraction, width=np.diff(bin_edges), alpha=0.7,
           color='coral', edgecolor='black', linewidth=0.5, align='edge', zorder=5)
    
    colors = ['red', 'orange', 'purple', 'green', 'brown']
    for i, threshold in enumerate(thresholds):
        if len(active_particles) > 0 and threshold <= active_particles.max():
            ax.axvline(threshold, color=colors[i % len(colors)], linestyle='--', linewidth=2,
                      label=f'Threshold {threshold:,}', zorder=10)
    
    ax.set_xlabel('Beached Particles per Cell (Monthly Total)', fontsize=12)
    ax.set_ylabel('Fraction of Active Cells', fontsize=12)
    ax.set_xscale('log')
    ax.set_title(f'Distribution of Beached Particle Counts - {month_name} 2024\n'
                f'({active_cells_baseline:,} active cells)', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def create_threshold_impact_plot(active_loss_stats, thresholds, month_name, active_cells_baseline):
    """Create threshold impact plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    active_kept = [active_loss_stats[t]['percentage_of_active_kept'] for t in thresholds]
    active_lost = [active_loss_stats[t]['percentage_of_active_lost'] for t in thresholds]
    
    ax.bar(range(len(thresholds)), active_kept, alpha=0.7, 
           color='darkgreen', label='Active Cells Kept')
    ax.bar(range(len(thresholds)), active_lost, bottom=active_kept,
           alpha=0.7, color='darkred', label='Active Cells Lost')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Active Cells', fontsize=12)
    ax.set_title(f'Impact of Thresholding on Active Cells - {month_name} 2024\n'
                f'(Baseline: {active_cells_baseline:,} active cells)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'{t:,}' for t in thresholds], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    for i, (kept, lost) in enumerate(zip(active_kept, active_lost)):
        if kept > 5:
            ax.text(i, kept/2, f'{kept:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white')
        if lost > 5:
            ax.text(i, kept + lost/2, f'{lost:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white')
    
    plt.tight_layout()
    return fig

def apply_threshold_to_frequency(ds, threshold):
    """Apply threshold to beaching data and recalculate frequency using fixed steps."""
    # Get total particle counts
    total_counts = ds['total_particle_counts'].values
    
    # Apply threshold: set cells below threshold to 0
    thresholded_counts = np.where(total_counts >= threshold, total_counts, 0)
    
    # Recalculate frequency based on daily counts with threshold
    daily_counts = ds['daily_beaching_counts'].values
    days_in_month = daily_counts.shape[0]
    
    # For each cell, count days where particles >= threshold
    thresholded_daily = np.where(daily_counts >= threshold, 1, 0)
    days_with_beaching = np.sum(thresholded_daily, axis=0)
    frequency = days_with_beaching.astype(float) / days_in_month
    
    return frequency, thresholded_counts

def get_colormap_for_basemap(basemap_choice):
    """Return appropriate colormap for selected basemap."""
    colormap_config = {
        'OpenStreetMap (Default)': 'hot_r',
        'OpenStreetMap (Shortbread)': 'YlOrRd',  # Lighter colormap for dark basemap
        'CartoDB Positron (Light)': 'thermal'  # cmocean thermal for light basemap
    }
    return colormap_config.get(basemap_choice, 'hot_r')

def get_basemap_tiles(basemap_choice):
    """Return tile URL and attribution for selected basemap."""
    basemap_config = {
        'OpenStreetMap (Default)': {
            'tiles': 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
            'attr': 'OpenStreetMap'
        },
        'OpenStreetMap (Shortbread)': {
            'tiles': 'https://tiles.stadiamaps.com/tiles/osm_bright/{z}/{x}/{y}{r}.png',
            'attr': 'Stadia Maps'
        },
        'CartoDB Positron (Light)': {
            'tiles': 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            'attr': 'CartoDB'
        }
    }
    return basemap_config.get(basemap_choice, basemap_config['OpenStreetMap (Default)'])

def create_folium_map(ds, threshold, month_name, basemap_choice):
    """Create interactive Folium map with beaching frequency as grid cells using fixed_step standardization."""
    
    # Apply threshold
    frequency, _ = apply_threshold_to_frequency(ds, threshold)
    
    # Get coordinates
    lons = ds['lon'].values
    lats = ds['lat'].values
    
    # Calculate map center
    lon_center = (lons.min() + lons.max()) / 2
    lat_center = (lats.min() + lats.max()) / 2
    
    # Get basemap configuration
    basemap_config = get_basemap_tiles(basemap_choice)
    
    # Create base map
    m = folium.Map(
        location=[lat_center, lon_center],
        zoom_start=6,
        tiles=basemap_config['tiles'],
        attr=basemap_config['attr'],
        control_scale=True
    )
    
    # Calculate grid cell size
    dlon = lons[1] - lons[0] if len(lons) > 1 else 0.05
    dlat = lats[1] - lats[0] if len(lats) > 1 else 0.05
    
    # Fixed step normalization (same as original script)
    discrete_levels = np.arange(0.0, 1.0 + 0.025, 0.025)
    
    # Get colormap
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import BoundaryNorm
    
    cmap_name = get_colormap_for_basemap(basemap_choice)
    
    # Handle cmocean colormaps
    if cmap_name == 'thermal':
        import cmocean
        cmap = cmocean.cm.thermal
    else:
        cmap = plt.get_cmap(cmap_name)
    
    norm = BoundaryNorm(discrete_levels, cmap.N)
    
    # Add grid cells as rectangles
    for i in range(len(lats)):
        for j in range(len(lons)):
            if frequency[i, j] > 0:
                # Get color from colormap using fixed step normalization
                color_val = norm(frequency[i, j])
                rgba = cmap(color_val)
                hex_color = '#{:02x}{:02x}{:02x}'.format(
                    int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
                )
                
                # Define rectangle bounds
                bounds = [
                    [lats[i] - dlat/2, lons[j] - dlon/2],
                    [lats[i] + dlat/2, lons[j] + dlon/2]
                ]
                
                # Add rectangle
                folium.Rectangle(
                    bounds=bounds,
                    color=hex_color,
                    fill=True,
                    fillColor=hex_color,
                    fillOpacity=0.7,
                    weight=0.5,
                    popup=f'Frequency: {frequency[i, j]:.3f}<br>Lat: {lats[i]:.2f}<br>Lon: {lons[j]:.2f}'
                ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add title
    title_html = f'''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: 90px; 
                background-color: white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px">
    <b>Beaching Frequency - {month_name} 2024</b><br>
    Threshold: ≥{threshold} particles<br>
    <small>Basemap: {basemap_choice}<br>
    Standardization: Fixed steps (0.025 intervals)</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    return m, cmap, norm

def create_colorbar_figure(cmap, norm, basemap_choice):
    """Create a standalone colorbar figure matching the map colors."""
    fig, ax = plt.subplots(figsize=(10, 1.5))
    
    # Create colorbar
    from matplotlib.colorbar import ColorbarBase
    
    tick_positions = [0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 0.90, 1.0]
    tick_labels = [f'{v:.2f}' for v in tick_positions]
    
    cb = ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')
    cb.set_ticks(tick_positions)
    cb.set_ticklabels(tick_labels)
    cb.set_label('Beaching Occurrence Frequency', fontsize=11)
    cb.ax.tick_params(labelsize=9)
    
    plt.tight_layout()
    return fig

# =============================================================================
# STREAMLIT APP
# =============================================================================

def main():
    st.title("🌊 Interactive Beaching Analysis Dashboard")
    st.markdown("---")
    
    # Load available months
    available_months = load_available_months()
    
    if not available_months:
        st.error(f"❌ No data files found in '{DATA_DIR}' directory!")
        st.info("Please run the 'convert_pkl_to_nc.py' script first to convert your pkl files to netCDF format.")
        return
    
    st.sidebar.header("📊 Available Data")
    st.sidebar.success(f"✅ {len(available_months)} months loaded")
    
    # Main tabs
    tab1, tab2 = st.tabs(["🔍 Threshold Testing", "🗺️ Frequency Maps"])
    
    # =============================================================================
    # TAB 1: THRESHOLD TESTING
    # =============================================================================
    
    with tab1:
        st.header("🔍 Threshold Testing & Histogram Analysis")
        st.markdown("Test different thresholds and analyze their impact on the dataset.")
        
        # Threshold input
        st.subheader("Define Thresholds")
        col1, col2 = st.columns([3, 1])
        
        with col1:
            threshold_input = st.text_input(
                "Enter thresholds (comma-separated)",
                value="1, 10, 50, 100, 500",
                help="Enter threshold values separated by commas"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            generate_histograms = st.button("🚀 Generate Histograms", type="primary", use_container_width=True)
        
        if generate_histograms:
            try:
                # Parse thresholds
                thresholds = [int(x.strip()) for x in threshold_input.split(',')]
                thresholds = sorted(thresholds)
                
                st.info(f"📊 Analyzing with thresholds: {thresholds}")
                
                # Progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Create containers for results
                results_container = st.container()
                
                with results_container:
                    # Process each month
                    for idx, (month_num, filepath) in enumerate(sorted(available_months.items())):
                        month_name = MONTH_INFO[month_num]['name']
                        
                        status_text.text(f"Processing {month_name}... ({idx+1}/{len(available_months)})")
                        progress_bar.progress((idx + 1) / len(available_months))
                        
                        # Load data
                        ds = load_month_data(filepath)
                        
                        # Get total particle counts
                        total_particles = ds['total_particle_counts'].values.flatten()
                        active_particles = total_particles[total_particles > 0]
                        
                        if len(active_particles) == 0:
                            st.warning(f"⚠️ {month_name}: No beaching data available")
                            continue
                        
                        # Calculate statistics
                        stats, active_loss_stats, active_cells_baseline = calculate_threshold_statistics(
                            total_particles, thresholds
                        )
                        
                        # Create section for this month
                        with st.expander(f"📅 {month_name} 2024", expanded=False):
                            # Display statistics
                            col1, col2, col3, col4 = st.columns(4)
                            col1.metric("Total Beached", f"{ds.attrs.get('total_beached_particles', 0):,}")
                            col2.metric("Active Cells", f"{active_cells_baseline:,}")
                            col3.metric("Max Beached/Cell", f"{total_particles.max():,}")
                            col4.metric("Mean (Active)", f"{active_particles.mean():.1f}")
                            
                            # Create plots
                            plot_col1, plot_col2 = st.columns(2)
                            
                            with plot_col1:
                                st.pyplot(create_distribution_histogram(
                                    active_particles, thresholds, month_name, active_cells_baseline
                                ))
                            
                            with plot_col2:
                                st.pyplot(create_threshold_impact_plot(
                                    active_loss_stats, thresholds, month_name, active_cells_baseline
                                ))
                            
                            # Threshold impact table
                            st.subheader("📊 Threshold Impact Summary")
                            
                            impact_data = []
                            for threshold in thresholds:
                                impact_data.append({
                                    'Threshold': f'≥{threshold}',
                                    'Cells Kept': stats[threshold]['active_cells'],
                                    '% Kept (Active)': f"{active_loss_stats[threshold]['percentage_of_active_kept']:.1f}%",
                                    'Cells Lost (Active)': active_loss_stats[threshold]['cells_lost_from_active'],
                                    '% Lost (Active)': f"{active_loss_stats[threshold]['percentage_of_active_lost']:.1f}%"
                                })
                            
                            st.dataframe(pd.DataFrame(impact_data), use_container_width=True, hide_index=True)
                
                status_text.text("✅ Analysis complete!")
                st.success("🎉 All histograms generated successfully!")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    # =============================================================================
    # TAB 2: FREQUENCY MAPS
    # =============================================================================
    
    with tab2:
        st.header("🗺️ Interactive Frequency Maps")
        st.markdown("Generate frequency maps with custom thresholds and explore interactively.")
        
        # Threshold and month selection
        col1, col2, col3 = st.columns([2, 2, 2])
        
        with col1:
            map_threshold = st.number_input(
                "Select Threshold",
                min_value=1,
                max_value=10000,
                value=50,
                step=10,
                help="Minimum number of beached particles to include in frequency calculation"
            )
        
        with col2:
            selected_month = st.selectbox(
                "Select Month",
                options=sorted(available_months.keys()),
                format_func=lambda x: MONTH_INFO[x]['name'],
                help="Choose which month to visualize"
            )
        
        with col3:
            basemap_choice = st.selectbox(
                "Select Basemap",
                options=['OpenStreetMap (Default)', 'OpenStreetMap (Shortbread)', 'CartoDB Positron (Light)'],
                help="Choose basemap style"
            )
        
        # Generate map button
        generate_map = st.button("🗺️ Generate Map", type="primary", use_container_width=True)
        
        if generate_map:
            month_name = MONTH_INFO[selected_month]['name']
            filepath = available_months[selected_month]
            
            with st.spinner(f"Generating interactive map for {month_name}..."):
                # Load data
                ds = load_month_data(filepath)
                
                # Display summary statistics
                frequency, thresholded_counts = apply_threshold_to_frequency(ds, map_threshold)
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Threshold", f"≥{map_threshold}")
                col2.metric("Active Cells", f"{np.sum(frequency > 0):,}")
                col3.metric("Max Frequency", f"{frequency.max():.3f}")
                col4.metric("Total Beached", f"{ds.attrs.get('total_beached_particles', 0):,}")
                
                st.markdown("---")
                
                # Generate and display map
                st.subheader(f"🗺️ Interactive Map - {month_name} 2024")
                st.markdown("*Zoom and pan to explore the beaching frequency distribution*")
                
                folium_map, cmap, norm = create_folium_map(ds, map_threshold, month_name, basemap_choice)
                
                # Display the map
                st.components.v1.html(folium_map._repr_html_(), height=600)
                
                # Display colorbar
                st.subheader("📊 Color Scale")
                colorbar_fig = create_colorbar_figure(cmap, norm, basemap_choice)
                st.pyplot(colorbar_fig)
                
                st.success(f"✅ Map generated for {month_name} with threshold ≥{map_threshold}")

# =============================================================================
# RUN APP
# =============================================================================

if __name__ == "__main__":
    main()