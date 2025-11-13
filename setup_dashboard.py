#!/usr/bin/env python3
# =============================================================================
# SETUP SCRIPT FOR BEACHING ANALYSIS DASHBOARD
# =============================================================================
# Automates the setup process: checks dependencies, converts data, and runs app
# =============================================================================

import os
import sys
import subprocess
import glob

def print_header(text):
    """Print formatted header."""
    print("\n" + "="*70)
    print(text)
    print("="*70 + "\n")

def check_python_version():
    """Check Python version."""
    print("🐍 Checking Python version...")
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required!")
        print(f"   Current version: {sys.version}")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    return True

def check_directories():
    """Check if required directories exist."""
    print("\n📁 Checking directories...")
    
    # Only check for NetCDF directory
    if os.path.exists('beaching_data_nc'):
        print(f"✅ beaching_data_nc/ - NetCDF data directory")
        return True
    else:
        os.makedirs('beaching_data_nc', exist_ok=True)
        print(f"✅ beaching_data_nc/ - Created (place your .nc files here)")
        return True

def check_nc_files():
    """Check for NetCDF files."""
    print("\n📦 Checking for NetCDF files...")
    
    nc_files = glob.glob("beaching_data_nc/*.nc")
    
    if not nc_files:
        print("❌ No NetCDF files found in beaching_data_nc/")
        print("   Please place your .nc files in the beaching_data_nc/ directory.")
        return False
    
    print(f"✅ Found {len(nc_files)} NetCDF file(s)")
    for nc_file in sorted(nc_files):
        print(f"   • {os.path.basename(nc_file)}")
    return True

def install_dependencies():
    """Install required packages."""
    print_header("📦 INSTALLING DEPENDENCIES")
    
    if not os.path.exists('requirements.txt'):
        print("❌ requirements.txt not found!")
        return False
    
    print("Installing packages from requirements.txt...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("\n✅ All dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("\n❌ Failed to install dependencies!")
        return False



def run_streamlit():
    """Run Streamlit app."""
    print_header("🚀 LAUNCHING STREAMLIT DASHBOARD")
    
    if not os.path.exists('streamlit_beaching_dashboard.py'):
        print("❌ streamlit_beaching_dashboard.py not found!")
        return False
    
    print("Starting Streamlit app...")
    print("The dashboard will open in your default web browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.check_call([sys.executable, "-m", "streamlit", "run", "streamlit_beaching_dashboard.py"])
    except KeyboardInterrupt:
        print("\n\n✅ Dashboard stopped.")
    except subprocess.CalledProcessError:
        print("\n❌ Failed to start Streamlit!")
        return False
    
    return True

def main():
    """Main setup function."""
    print_header("🌊 BEACHING ANALYSIS DASHBOARD - SETUP")
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check directories
    if not check_directories():
        print("\n❌ Setup failed: Directory check failed")
        sys.exit(1)
    
    # Step 3: Check for NetCDF files
    has_nc = check_nc_files()
    
    if not has_nc:
        print("\n❌ Setup incomplete: No NetCDF files found")
        print("\n📝 Please place your .nc files in beaching_data_nc/ directory")
        print("   Expected files: beaching_01_january.nc through beaching_12_december.nc")
        proceed = input("\n❓ Do you want to continue to install dependencies anyway? (y/n): ").lower().strip() == 'y'
        if not proceed:
            sys.exit(0)
    
    # Step 4: Install dependencies
    need_install = input("\n❓ Install/update dependencies? (y/n): ").lower().strip() == 'y'
    
    if need_install:
        if not install_dependencies():
            sys.exit(1)
    
    # Final check
    if not has_nc:
        print("\n⚠️  Warning: No NetCDF files available!")
        print("   The dashboard will not work without data files.")
        print("   Please add .nc files to beaching_data_nc/ and restart.")
        sys.exit(0)
    
    # Launch Streamlit
    print("\n✅ Setup complete!")
    launch = input("\n❓ Launch Streamlit dashboard now? (y/n): ").lower().strip() == 'y'
    
    if launch:
        run_streamlit()
    else:
        print("\n✅ Setup complete! You can launch the dashboard anytime with:")
        print("   streamlit run streamlit_beaching_dashboard.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Setup interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        sys.exit(1)
