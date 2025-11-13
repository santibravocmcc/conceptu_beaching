# =============================================================================
# BEACHED PARTICLE COUNT HISTOGRAM ANALYSIS - SIMPLIFIED
# =============================================================================
# Creates distribution histogram and threshold impact analysis for beaching data
# =============================================================================

import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import glob
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

# Target month
TARGET_MONTH = "JANUARY"
TARGET_MONTH_NUM = 1

# Directory containing intermediate beaching count files
INTERMEDIATE_DIR = "intermediate_beaching_counts"

# Results directory
RESULTS_DIR = "results_beaching_histograms"

# Create results directory
os.makedirs(RESULTS_DIR, exist_ok=True)

# Thresholds to analyze (adjust based on your beaching data scale)
PARTICLE_THRESHOLDS = [1, 10, 50, 100, 500]

# Resolution to analyze
ORIGINAL_RESOLUTION = 0.05

# Histogram parameters
N_BINS = 20

# =============================================================================
# MONTH INFORMATION
# =============================================================================

MONTH_INFO = {
    "JANUARY": {"name": "January", "days": 31, "month_num": 1},
    "FEBRUARY": {"name": "February", "days": 28, "month_num": 2},
    "MARCH": {"name": "March", "days": 31, "month_num": 3},
    "APRIL": {"name": "April", "days": 30, "month_num": 4},
    "MAY": {"name": "May", "days": 31, "month_num": 5},
    "JUNE": {"name": "June", "days": 30, "month_num": 6},
    "JULY": {"name": "July", "days": 31, "month_num": 7},
    "AUGUST": {"name": "August", "days": 31, "month_num": 8},
    "SEPTEMBER": {"name": "September", "days": 30, "month_num": 9},
    "OCTOBER": {"name": "October", "days": 31, "month_num": 10},
    "NOVEMBER": {"name": "November", "days": 30, "month_num": 11},
    "DECEMBER": {"name": "December", "days": 31, "month_num": 12}
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def find_beaching_files(target_month_num, grid_resolution):
    """Find all beaching count files for target month and resolution."""
    res_str = f"{int(grid_resolution * 100):03d}"
    pattern = f"beaching_count_{target_month_num:02d}_{res_str}.pkl"
    
    files = []
    
    # Check month subdirectory
    target_month_name = None
    for month_key, month_data in MONTH_INFO.items():
        if month_data['month_num'] == target_month_num:
            target_month_name = month_key
            break
    
    if target_month_name:
        subdir_path = os.path.join(INTERMEDIATE_DIR, target_month_name)
        if os.path.exists(subdir_path):
            files.extend(glob.glob(os.path.join(subdir_path, pattern)))
    
    # Also check root directory
    files.extend(glob.glob(os.path.join(INTERMEDIATE_DIR, pattern)))
    
    return sorted(list(set(files)))

def load_beaching_file(filepath):
    """Load data from a beaching file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    if data.get('data_type') != 'beaching_frequency':
        print(f"WARNING: File {filepath} may not be a beaching frequency file!")
    
    return data

def calculate_threshold_statistics(total_particles_per_cell, thresholds, total_cells):
    """Calculate statistics for each threshold."""
    stats = {}
    
    for threshold in thresholds:
        active_cells = np.sum(total_particles_per_cell >= threshold)
        inactive_cells = total_cells - active_cells
        
        stats[threshold] = {
            'active_cells': active_cells,
            'inactive_cells': inactive_cells,
            'percentage_kept': (active_cells / total_cells) * 100 if total_cells > 0 else 0,
            'percentage_lost': (inactive_cells / total_cells) * 100 if total_cells > 0 else 0
        }
    
    return stats

def create_distribution_histogram(active_particles, thresholds, target_month_name, 
                                  output_path, active_cells_baseline):
    """Create distribution histogram of beached particle counts."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create bins
    if len(active_particles) > 0:
        bins = np.logspace(0, np.log10(active_particles.max()), N_BINS)
    else:
        bins = [0, 1]
    
    # Create histogram
    counts, bin_edges, patches = ax.hist(active_particles, bins=bins, alpha=0.7, 
                                         color='coral', edgecolor='black', linewidth=0.5,
                                         density=False)
    
    # Convert counts to fraction of active cells
    if active_cells_baseline > 0:
        counts_fraction = counts / active_cells_baseline
    else:
        counts_fraction = counts
    
    # Clear and replot with fractions
    ax.clear()
    ax.bar(bin_edges[:-1], counts_fraction, width=np.diff(bin_edges), alpha=0.7,
           color='coral', edgecolor='black', linewidth=0.5, align='edge', zorder=5)
    
    # Add threshold lines
    colors = ['red', 'orange', 'purple', 'green', 'brown']
    for i, threshold in enumerate(thresholds):
        if len(active_particles) > 0 and threshold <= active_particles.max():
            ax.axvline(threshold, color=colors[i % len(colors)], linestyle='--', linewidth=2,
                      label=f'Threshold {threshold:,}', zorder=10)
    
    ax.set_xlabel('Beached Particles per Cell (Monthly Total)', fontsize=12)
    ax.set_ylabel('Fraction of Active Cells', fontsize=12)
    ax.set_xscale('log')
    ax.set_title(f'Distribution of Beached Particle Counts per Cell - {target_month_name} 2024\n'
                f'(Fraction of {active_cells_baseline:,} active cells, log X-axis)', 
                fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved distribution histogram: {os.path.basename(output_path)}")
    plt.close()

def create_threshold_impact_plot(active_loss_stats, thresholds, target_month_name,
                                 output_path, active_cells_baseline):
    """Create threshold impact plot for active cells only."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Get percentages
    active_kept = [active_loss_stats[t]['percentage_of_active_kept'] for t in thresholds]
    active_lost = [active_loss_stats[t]['percentage_of_active_lost'] for t in thresholds]
    
    # Create stacked bar chart
    ax.bar(range(len(thresholds)), active_kept, alpha=0.7, 
           color='darkgreen', label='Active Cells Kept')
    ax.bar(range(len(thresholds)), active_lost, bottom=active_kept,
           alpha=0.7, color='darkred', label='Active Cells Lost')
    
    ax.set_xlabel('Threshold', fontsize=12)
    ax.set_ylabel('Percentage of Active Cells', fontsize=12)
    ax.set_title(f'Impact of Thresholding on ACTIVE Cells Only - {target_month_name} 2024\n'
                f'(Baseline: {active_cells_baseline:,} active cells)',
                fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'{t:,}' for t in thresholds], rotation=45)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add percentage labels
    for i, (kept, lost) in enumerate(zip(active_kept, active_lost)):
        if kept > 5:
            ax.text(i, kept/2, f'{kept:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white')
        if lost > 5:
            ax.text(i, kept + lost/2, f'{lost:.1f}%', ha='center', va='center', 
                   fontweight='bold', color='white')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved threshold impact plot: {os.path.basename(output_path)}")
    plt.close()

# =============================================================================
# MAIN ANALYSIS FUNCTION
# =============================================================================

def analyze_beaching_histograms():
    """Main function to analyze beached particle count histograms."""
    
    print("="*70)
    print("BEACHED PARTICLE COUNT HISTOGRAM ANALYSIS")
    print("="*70)
    print(f"Target Month: {MONTH_INFO[TARGET_MONTH]['name']}")
    print(f"Resolution: {ORIGINAL_RESOLUTION}°")
    print(f"Thresholds: {PARTICLE_THRESHOLDS}")
    print(f"Output Directory: {RESULTS_DIR}")
    
    # Find beaching files
    files_to_process = find_beaching_files(TARGET_MONTH_NUM, ORIGINAL_RESOLUTION)
    
    if not files_to_process:
        print(f"\n❌ No beaching files found for {TARGET_MONTH}!")
        print(f"Looking in: {INTERMEDIATE_DIR}")
        print("Please run Script 1B_beaching_frequency_OPTIMIZED.py first.")
        return
    
    print(f"\n📊 Found {len(files_to_process)} file(s)...")
    
    # Load beaching data
    beaching_data = None
    
    for i, filepath in enumerate(files_to_process, 1):
        filename = os.path.basename(filepath)
        print(f"[{i}/{len(files_to_process)}] Loading: {filename}")
        
        try:
            data = load_beaching_file(filepath)
            beaching_data = data
            
            # Display basic info
            print(f"  ✓ Simulations processed: {data.get('simulations_processed', 'N/A')}")
            print(f"  ✓ Total beached particles: {data.get('total_beached_particles', 0):,}")
            print(f"  ✓ Cells with beaching: {data.get('cells_with_beaching', 0):,}")
            print(f"  ✓ Max particle count: {data.get('max_particle_count', 0):,}")
            
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            continue
    
    if beaching_data is None:
        print("\n❌ No files were successfully processed!")
        return
    
    # Extract the total particle counts per cell
    total_particles_per_cell = beaching_data['total_particle_counts'].flatten()
    
    # Filter out cells with zero particles
    active_particles = total_particles_per_cell[total_particles_per_cell > 0]
    total_cells = len(total_particles_per_cell)
    zero_cells = total_cells - len(active_particles)
    active_cells_baseline = len(active_particles)
    
    print(f"\n📈 ANALYZING BEACHING DISTRIBUTION...")
    print(f"  • Total cells: {total_cells:,}")
    print(f"  • Active cells (with beaching): {active_cells_baseline:,}")
    print(f"  • Empty cells: {zero_cells:,}")
    print(f"  • Max beached in cell: {total_particles_per_cell.max():,}")
    if active_cells_baseline > 0:
        print(f"  • Mean beached per active cell: {active_particles.mean():.1f}")
        print(f"  • Median beached per active cell: {np.median(active_particles):.1f}")
    
    # Calculate threshold statistics
    threshold_stats = calculate_threshold_statistics(total_particles_per_cell, 
                                                     PARTICLE_THRESHOLDS, total_cells)
    
    # Calculate active cell loss statistics
    active_loss_stats = {}
    for threshold in PARTICLE_THRESHOLDS:
        cells_kept = threshold_stats[threshold]['active_cells']
        cells_lost_from_active = active_cells_baseline - cells_kept
        percentage_of_active_kept = (cells_kept / active_cells_baseline) * 100 if active_cells_baseline > 0 else 0
        percentage_of_active_lost = (cells_lost_from_active / active_cells_baseline) * 100 if active_cells_baseline > 0 else 0
        
        active_loss_stats[threshold] = {
            'cells_kept_from_active': cells_kept,
            'cells_lost_from_active': cells_lost_from_active,
            'percentage_of_active_kept': percentage_of_active_kept,
            'percentage_of_active_lost': percentage_of_active_lost
        }
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create distribution histogram
    dist_filename = f"beaching_distribution_{TARGET_MONTH.lower()}_{timestamp}.png"
    dist_path = os.path.join(RESULTS_DIR, dist_filename)
    create_distribution_histogram(active_particles, PARTICLE_THRESHOLDS, 
                                  MONTH_INFO[TARGET_MONTH]['name'], 
                                  dist_path, active_cells_baseline)
    
    # Create threshold impact plot
    threshold_filename = f"beaching_threshold_impact_{TARGET_MONTH.lower()}_{timestamp}.png"
    threshold_path = os.path.join(RESULTS_DIR, threshold_filename)
    create_threshold_impact_plot(active_loss_stats, PARTICLE_THRESHOLDS,
                                 MONTH_INFO[TARGET_MONTH]['name'],
                                 threshold_path, active_cells_baseline)
    
    # Create detailed threshold report (text file only)
    report_filename = f"beaching_threshold_report_{TARGET_MONTH.lower()}_{timestamp}.txt"
    report_path = os.path.join(RESULTS_DIR, report_filename)
    
    with open(report_path, 'w') as f:
        f.write("BEACHED PARTICLE COUNT THRESHOLD ANALYSIS REPORT\n")
        f.write("="*70 + "\n\n")
        f.write(f"Month: {MONTH_INFO[TARGET_MONTH]['name']} 2024\n")
        f.write(f"Simulations: {beaching_data.get('simulations_processed', 'N/A')}\n")
        f.write(f"Total beached particles: {beaching_data.get('total_beached_particles', 0):,}\n")
        f.write(f"Resolution: {ORIGINAL_RESOLUTION}°\n\n")
        
        f.write("="*70 + "\n")
        f.write("DATASET STATISTICS\n")
        f.write("="*70 + "\n\n")
        f.write(f"Total Grid Cells: {total_cells:,}\n")
        f.write(f"Active Cells (with beaching): {active_cells_baseline:,} ({active_cells_baseline/total_cells*100:.1f}%)\n")
        f.write(f"Empty Cells: {zero_cells:,} ({zero_cells/total_cells*100:.1f}%)\n\n")
        
        f.write("Active Cells Statistics:\n")
        f.write(f"  • Min beached/cell: {active_particles.min() if len(active_particles) > 0 else 0:,}\n")
        f.write(f"  • Max beached/cell: {active_particles.max() if len(active_particles) > 0 else 0:,}\n")
        f.write(f"  • Mean beached/cell: {active_particles.mean() if len(active_particles) > 0 else 0:.1f}\n")
        f.write(f"  • Median beached/cell: {np.median(active_particles) if len(active_particles) > 0 else 0:.1f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("THRESHOLD IMPACT ANALYSIS (vs Total Grid)\n")
        f.write("="*70 + "\n\n")
        for threshold in PARTICLE_THRESHOLDS:
            stats = threshold_stats[threshold]
            f.write(f"Threshold ≥{threshold:,} particles:\n")
            f.write(f"  • Cells kept: {stats['active_cells']:,} ({stats['percentage_kept']:.2f}%)\n")
            f.write(f"  • Cells lost: {stats['inactive_cells']:,} ({stats['percentage_lost']:.2f}%)\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"ACTIVE CELL LOSS ANALYSIS (vs Threshold=1 baseline: {active_cells_baseline:,} cells)\n")
        f.write("="*70 + "\n\n")
        for threshold in PARTICLE_THRESHOLDS[1:]:  # Skip threshold=1
            active_stats = active_loss_stats[threshold]
            f.write(f"Threshold ≥{threshold:,} particles:\n")
            f.write(f"  • Active cells kept: {active_stats['cells_kept_from_active']:,} ({active_stats['percentage_of_active_kept']:.2f}%)\n")
            f.write(f"  • Active cells lost: {active_stats['cells_lost_from_active']:,} ({active_stats['percentage_of_active_lost']:.2f}%)\n\n")
        
        f.write("="*70 + "\n")
        f.write(f"Processing completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"✅ Saved report: {os.path.basename(report_path)}")
    
    # Summary
    print("\n" + "="*70)
    print("📊 THRESHOLD IMPACT SUMMARY:")
    print("="*70)
    print("\nTOTAL GRID IMPACT:")
    for threshold in PARTICLE_THRESHOLDS:
        stats = threshold_stats[threshold]
        print(f"  Threshold ≥{threshold:,}: Keep {stats['percentage_kept']:.1f}%, Lose {stats['percentage_lost']:.1f}%")
    
    print(f"\nACTIVE CELLS IMPACT (vs {active_cells_baseline:,} active cells):")
    for threshold in PARTICLE_THRESHOLDS[1:]:
        active_stats = active_loss_stats[threshold]
        print(f"  Threshold ≥{threshold:,}: Keep {active_stats['percentage_of_active_kept']:.1f}%, Lose {active_stats['percentage_of_active_lost']:.1f}%")
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Results saved to: {RESULTS_DIR}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    analyze_beaching_histograms()
