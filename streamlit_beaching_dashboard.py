# =============================================================================
# PARTICLE ANALYSIS DASHBOARD
# =============================================================================
# Streamlit application for exploring beached and surface particle data
# =============================================================================

import os
import glob
import json
import base64

import numpy as np
import xarray as xr
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import folium
from folium.raster_layers import ImageOverlay
import cmocean
import streamlit as st

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Particle Analysis Dashboard",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =============================================================================
# CONFIGURATION
# =============================================================================

BEACHED_DIR = "beaching_data_nc"
SURFACE_DIR = "surface_data_zenodo_nc"
MPA_GEOJSON = "protected_area/mpa_mediterranean.geojson"
GA_MEASUREMENT_ID = "G-G54Y1MVWSJ"

LOGO_DIR = "logo"
LOGOS = {
    "conceptu": {"normal": "conceptu_normal.png", "dark": "conceptu_dark.png",
                 "link": "https://www.lifeconceptu.eu/en/", "alt": "LIFE CONCEPTU MARIS"},
    "cmcc": {"normal": "cmcc_normal.png", "dark": "cmcc_dark.png",
             "link": "https://www.cmcc.it/", "alt": "CMCC"},
}

PORTFOLIO_URL = "https://santibravocmcc.github.io/portfolio/"
WDPA_URL = "https://www.protectedplanet.net"

N_BINS = 20

MONTH_INFO = {
    1: {"name": "January", "abbr": "JAN"},
    2: {"name": "February", "abbr": "FEB"},
    3: {"name": "March", "abbr": "MAR"},
    4: {"name": "April", "abbr": "APR"},
    5: {"name": "May", "abbr": "MAY"},
    6: {"name": "June", "abbr": "JUN"},
    7: {"name": "July", "abbr": "JUL"},
    8: {"name": "August", "abbr": "AUG"},
    9: {"name": "September", "abbr": "SEP"},
    10: {"name": "October", "abbr": "OCT"},
    11: {"name": "November", "abbr": "NOV"},
    12: {"name": "December", "abbr": "DEC"},
}

# Map styles: tile layer + colormap + protected-area outline/fill tuned per basemap
MAP_STYLES = {
    "Light": {"tiles": "CartoDB positron", "attr": None, "cmap": "thermal",
              "mpa_stroke": "#1b5e20", "mpa_fill": "#43a047"},
    "Dark": {"tiles": "CartoDB dark_matter", "attr": None, "cmap": "viridis",
             "mpa_stroke": "#69f0ae", "mpa_fill": "#69f0ae"},
    "Streets": {"tiles": "OpenStreetMap", "attr": None, "cmap": "hot_r",
                "mpa_stroke": "#1b5e20", "mpa_fill": "#2e7d32"},
}

# =============================================================================
# STYLING
# =============================================================================

THEMES = {
    "light": {
        "mode": "light",
        "bg": "#ffffff", "surface": "#ffffff", "input_bg": "#f3f6f8",
        "text": "#1a2b34", "muted": "#5a727d", "border": "#e3eaee",
        "heading": "#0f3d4d", "primary": "#1f6f8b", "primary_hover": "#155366",
        "primary_text": "#ffffff", "shadow": "rgba(15, 61, 77, 0.08)",
        "nav_bg": "#f3f6f8", "nav_text": "#5a727d", "nav_hover": "#e3eaee",
        "nav_sel_bg": "#1f6f8b", "nav_sel_text": "#ffffff",
        "subnav_sel_bg": "#e3eaee", "subnav_sel_text": "#0f3d4d",
        "plot_face": "#ffffff", "plot_text": "#1a2b34", "plot_grid": "#cdd8de",
        "bar_text": "#ffffff",
    },
    "dark": {
        "mode": "dark",
        "bg": "#0e1b22", "surface": "#16252e", "input_bg": "#1b2c36",
        "text": "#e6eef2", "muted": "#9fb3bd", "border": "#273a45",
        "heading": "#7fd1e0", "primary": "#2a93a8", "primary_hover": "#37a9bf",
        "primary_text": "#06141a", "shadow": "rgba(0, 0, 0, 0.45)",
        "nav_bg": "#16252e", "nav_text": "#9fb3bd", "nav_hover": "#1f3540",
        "nav_sel_bg": "#2a93a8", "nav_sel_text": "#06141a",
        "subnav_sel_bg": "#273a45", "subnav_sel_text": "#e6eef2",
        "plot_face": "#16252e", "plot_text": "#e6eef2", "plot_grid": "#3a4f5a",
        "bar_text": "#ffffff",
    },
}


def inject_css(t):
    """Inject the active theme: typography, layout, colours, hidden chrome."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Source+Serif+4:wght@600;700&display=swap');

        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header[data-testid="stHeader"] {{display: none;}}
        div[data-testid="stToolbar"] {{display: none;}}
        div[data-testid="stDecoration"] {{display: none;}}

        .stApp, [data-testid="stAppViewContainer"] {{ background: {t['bg']}; }}
        html, body, [class*="css"], .stMarkdown, p, span, li {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            color: {t['text']};
        }}

        .block-container {{
            padding-top: 1.2rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }}

        .hero {{
            background: linear-gradient(135deg, #0f3d4d 0%, #1f6f8b 55%, #2a93a8 100%);
            border-radius: 16px;
            padding: 2.4rem 2.6rem;
            margin-bottom: 1.6rem;
            color: #ffffff;
            box-shadow: 0 10px 30px {t['shadow']};
        }}
        .hero h1 {{
            font-family: 'Source Serif 4', Georgia, serif;
            font-size: 2.1rem; font-weight: 700; margin: 0;
            color: #ffffff !important; letter-spacing: -0.01em;
        }}
        .hero p {{
            margin: 0.5rem 0 0 0; font-size: 1.02rem;
            color: rgba(255, 255, 255, 0.82) !important; font-weight: 400;
        }}

        h1, h2, h3 {{
            font-family: 'Source Serif 4', Georgia, serif;
            letter-spacing: -0.01em; color: {t['heading']};
        }}

        div[data-testid="stMetric"] {{
            background: {t['surface']};
            border: 1px solid {t['border']};
            border-radius: 12px; padding: 1rem 1.2rem;
            box-shadow: 0 2px 8px {t['shadow']};
        }}
        div[data-testid="stMetric"] label, div[data-testid="stMetric"] label p {{
            color: {t['muted']}; font-weight: 500; font-size: 0.8rem;
            text-transform: uppercase; letter-spacing: 0.04em;
        }}
        div[data-testid="stMetricValue"] {{ color: {t['heading']}; font-weight: 700; }}

        div[data-testid="stButton"] > button {{
            border-radius: 10px; border: none;
            background: {t['primary']}; color: {t['primary_text']};
            font-weight: 600; padding: 0.55rem 1.4rem;
            transition: background 0.15s ease;
        }}
        div[data-testid="stButton"] > button:hover {{ background: {t['primary_hover']}; }}
        div[data-testid="stButton"] > button p {{ color: {t['primary_text']}; }}

        /* Inputs */
        div[data-baseweb="input"], div[data-baseweb="select"] > div,
        div[data-baseweb="base-input"], div[data-baseweb="input"] > div {{
            border-radius: 10px;
            background-color: {t['input_bg']} !important;
            border-color: {t['border']} !important;
        }}
        div[data-baseweb="input"] input, div[data-baseweb="select"] div,
        textarea, div[data-baseweb="select"] span {{ color: {t['text']} !important; }}

        /* Expander */
        div[data-testid="stExpander"] {{
            border: 1px solid {t['border']}; border-radius: 12px;
            background: {t['surface']};
        }}
        div[data-testid="stExpander"] summary p {{ color: {t['heading']}; font-weight: 600; }}

        /* Dataframe */
        div[data-testid="stDataFrame"] {{
            border-radius: 12px; overflow: hidden; border: 1px solid {t['border']};
        }}

        /* Tabs */
        button[data-baseweb="tab"] {{
            font-size: 1rem; font-weight: 500; color: {t['nav_text']};
            padding: 0.4rem 0.2rem;
        }}
        button[data-baseweb="tab"] p {{ font-size: 1rem; font-weight: 500; color: {t['nav_text']}; }}
        button[data-baseweb="tab"][aria-selected="true"] p {{ color: {t['heading']}; font-weight: 600; }}
        div[data-baseweb="tab-highlight"] {{ background-color: {t['primary']}; }}
        div[data-baseweb="tab-border"] {{ background-color: {t['border']}; }}
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {{ gap: 1.6rem; }}

        /* Toggle label colour */
        div[data-testid="stToggle"] label p, label[data-baseweb="checkbox"] {{ color: {t['muted']}; font-weight: 500; }}

        .muted {{ color: {t['muted']}; font-size: 0.92rem; }}
        hr {{ border-color: {t['border']}; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title, subtitle):
    st.markdown(
        f"""<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>""",
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def image_data_uri(path):
    """Return a base64 data URI for a local image (so it can go in HTML)."""
    if not os.path.exists(path):
        return ""
    ext = os.path.splitext(path)[1].lstrip(".").lower()
    mime = "image/webp" if ext == "webp" else f"image/{'jpeg' if ext in ('jpg', 'jpeg') else ext}"
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def render_logos(t):
    """Clickable partner logos, right-aligned; version follows the theme."""
    variant = "dark" if t["mode"] == "dark" else "normal"
    items = []
    for key in ("conceptu", "cmcc"):  # conceptu first, cmcc on the far right
        cfg = LOGOS[key]
        uri = image_data_uri(os.path.join(LOGO_DIR, cfg[variant]))
        if not uri:
            continue
        height = 52 if key == "conceptu" else 40
        items.append(
            f'<a href="{cfg["link"]}" target="_blank" rel="noopener" title="{cfg["alt"]}">'
            f'<img src="{uri}" alt="{cfg["alt"]}" style="height:{height}px;width:auto;display:block;"></a>'
        )
    st.markdown(
        '<div style="display:flex;justify-content:flex-end;align-items:center;gap:26px;'
        'min-height:54px;">' + "".join(items) + "</div>",
        unsafe_allow_html=True,
    )


def render_footer(t):
    """Subtle attribution + data citation at the bottom of the page."""
    st.markdown(
        f"""
        <hr style="margin-top:2.4rem;margin-bottom:1rem;border:none;border-top:1px solid {t['border']};">
        <div style="font-size:0.82rem;color:{t['muted']};line-height:1.6;padding-bottom:1.5rem;">
            Developed by
            <a href="{PORTFOLIO_URL}" target="_blank" rel="noopener"
               style="color:{t['primary']};text-decoration:none;font-weight:600;">Santiago Bravo</a>,
            CMCC Foundation–Euro-Mediterranean Center on Climate Change, Italy.
            <br>
            Protected-area data: UNEP-WCMC and IUCN (2026), Protected Planet: The World Database on
            Protected Areas (WDPA) and World Database on Other Effective Area-based Conservation
            Measures (WD-OECM) [Online], June 2026, Cambridge, UK: UNEP-WCMC and IUCN. Available at:
            <a href="{WDPA_URL}" target="_blank" rel="noopener"
               style="color:{t['primary']};text-decoration:none;">www.protectedplanet.net</a>.
        </div>
        """,
        unsafe_allow_html=True,
    )


def inject_analytics(measurement_id):
    """Load the Google Analytics 4 tag on the real app page.

    st.components.v1.html renders into a sandboxed child iframe, so a plain gtag
    snippet placed there is invisible to GA and to Google's tag detector (both
    only inspect the top-level document). The iframe is same-origin (srcdoc), so
    we reach up into the parent document and inject the tag there instead, once."""
    st.components.v1.html(
        f"""
        <script>
        (function() {{
            var doc = window.parent.document;
            if (doc.getElementById('ga-gtag-js')) return;  // inject only once
            var s = doc.createElement('script');
            s.id = 'ga-gtag-js';
            s.async = true;
            s.src = 'https://www.googletagmanager.com/gtag/js?id={measurement_id}';
            doc.head.appendChild(s);
            var s2 = doc.createElement('script');
            s2.innerHTML = "window.dataLayer = window.dataLayer || [];"
                + "function gtag(){{dataLayer.push(arguments);}}"
                + "gtag('js', new Date());"
                + "gtag('config', '{measurement_id}');";
            doc.head.appendChild(s2);
        }})();
        </script>
        """,
        height=0,
    )


def render_table(df, t):
    """Render a dataframe as a themed HTML table (st.dataframe's canvas grid
    can't follow the runtime dark/light theme, so we draw our own)."""
    header = "".join(
        f'<th style="text-align:left;padding:11px 16px;color:{t["muted"]};font-weight:600;'
        f'font-size:0.78rem;text-transform:uppercase;letter-spacing:0.04em;'
        f'border-bottom:1px solid {t["border"]};">{c}</th>'
        for c in df.columns
    )
    rows = ""
    for _, r in df.iterrows():
        cells = "".join(
            f'<td style="padding:11px 16px;color:{t["text"]};border-bottom:1px solid {t["border"]};">{v}</td>'
            for v in r
        )
        rows += f"<tr>{cells}</tr>"
    st.markdown(
        f'<div style="border:1px solid {t["border"]};border-radius:12px;overflow:hidden;'
        f'background:{t["surface"]};margin-top:0.4rem;">'
        f'<table style="width:100%;border-collapse:collapse;font-family:Inter,sans-serif;font-size:0.92rem;">'
        f"<thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></div>",
        unsafe_allow_html=True,
    )


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_beached_months():
    """Map month number -> filepath for beached particle data."""
    files = glob.glob(os.path.join(BEACHED_DIR, "beaching_*.nc"))
    months = {}
    for fp in files:
        name = os.path.basename(fp)
        months[int(name.split("_")[1])] = fp
    return months


@st.cache_data(show_spinner=False)
def load_surface_months():
    """Map month number -> filepath for surface particle data."""
    files = glob.glob(os.path.join(SURFACE_DIR, "surface_frequency_*.nc"))
    months = {}
    for fp in files:
        name = os.path.basename(fp)
        # surface_frequency_2024-01.nc -> 01
        month_num = int(name.split("-")[-1].split(".")[0])
        months[month_num] = fp
    return months


@st.cache_data(show_spinner=False)
def load_dataset(filepath):
    return xr.open_dataset(filepath)


@st.cache_data(show_spinner=False)
def load_mpa_geojson():
    """Raw GeoJSON text for the Marine Protected Areas layer ('' if missing)."""
    if not os.path.exists(MPA_GEOJSON):
        return ""
    with open(MPA_GEOJSON, "r") as f:
        return f.read()


# =============================================================================
# ANALYSIS (BEACHED ONLY)
# =============================================================================

def calculate_threshold_statistics(total_particles_per_cell, thresholds):
    total_cells = len(total_particles_per_cell)
    active_cells_baseline = int(np.sum(total_particles_per_cell > 0))

    stats, active_loss_stats = {}, {}
    for threshold in thresholds:
        active_cells = int(np.sum(total_particles_per_cell >= threshold))
        inactive_cells = total_cells - active_cells
        stats[threshold] = {
            "active_cells": active_cells,
            "inactive_cells": inactive_cells,
            "percentage_kept": (active_cells / total_cells * 100) if total_cells else 0,
            "percentage_lost": (inactive_cells / total_cells * 100) if total_cells else 0,
        }
        cells_lost = active_cells_baseline - active_cells
        active_loss_stats[threshold] = {
            "cells_kept_from_active": active_cells,
            "cells_lost_from_active": cells_lost,
            "percentage_of_active_kept": (active_cells / active_cells_baseline * 100) if active_cells_baseline else 0,
            "percentage_of_active_lost": (cells_lost / active_cells_baseline * 100) if active_cells_baseline else 0,
        }
    return stats, active_loss_stats, active_cells_baseline


def _style_axes(fig, ax, t):
    """Apply theme colours to a matplotlib figure."""
    fig.patch.set_facecolor(t["plot_face"])
    ax.set_facecolor(t["plot_face"])
    ax.tick_params(colors=t["plot_text"], labelcolor=t["plot_text"])
    ax.xaxis.label.set_color(t["plot_text"])
    ax.yaxis.label.set_color(t["plot_text"])
    ax.title.set_color(t["heading"])
    for spine in ax.spines.values():
        spine.set_color(t["plot_grid"])
    ax.grid(True, alpha=0.25, color=t["plot_grid"])
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_color(t["plot_text"])


def create_distribution_histogram(active_particles, thresholds, month_name, active_cells_baseline, t):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.logspace(0, np.log10(active_particles.max()), N_BINS) if len(active_particles) else [0, 1]
    counts, bin_edges = np.histogram(active_particles, bins=bins)
    counts_fraction = counts / active_cells_baseline if active_cells_baseline else counts

    ax.bar(bin_edges[:-1], counts_fraction, width=np.diff(bin_edges), align="edge",
           color="#2a93a8", edgecolor="white", linewidth=0.4, alpha=0.9, zorder=5)

    palette = ["#d1495b", "#edae49", "#8338ec", "#2a9d8f", "#9b5de5"]
    for i, threshold in enumerate(thresholds):
        if len(active_particles) and threshold <= active_particles.max():
            ax.axvline(threshold, color=palette[i % len(palette)], linestyle="--",
                       linewidth=1.6, label=f"≥ {threshold:,}", zorder=10)

    ax.set_xlabel("Beached particles per cell (monthly total)", fontsize=10)
    ax.set_ylabel("Fraction of active cells", fontsize=10)
    ax.set_xscale("log")
    ax.set_title(f"Distribution — {month_name} 2024  ·  {active_cells_baseline:,} active cells",
                 fontsize=11, fontweight="bold")
    ax.legend(frameon=False, fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    _style_axes(fig, ax, t)
    plt.tight_layout()
    return fig


def create_threshold_impact_plot(active_loss_stats, thresholds, month_name, active_cells_baseline, t):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    kept = [active_loss_stats[t]["percentage_of_active_kept"] for t in thresholds]
    lost = [active_loss_stats[t]["percentage_of_active_lost"] for t in thresholds]
    x = range(len(thresholds))

    ax.bar(x, kept, color="#1f6f8b", label="Active cells kept")
    ax.bar(x, lost, bottom=kept, color="#d1495b", label="Active cells lost")

    ax.set_xlabel("Threshold", fontsize=10)
    ax.set_ylabel("Percentage of active cells", fontsize=10)
    ax.set_title(f"Thresholding impact — {month_name} 2024",
                 fontsize=11, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{thr:,}" for thr in thresholds], rotation=45, fontsize=9)
    ax.legend(frameon=False, fontsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    for i, (k, l) in enumerate(zip(kept, lost)):
        if k > 6:
            ax.text(i, k / 2, f"{k:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
        if l > 6:
            ax.text(i, k + l / 2, f"{l:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
    _style_axes(fig, ax, t)
    ax.grid(True, alpha=0.25, axis="y", color=t["plot_grid"])
    plt.tight_layout()
    return fig


def apply_threshold_to_frequency(ds, threshold):
    """Recompute beaching frequency after applying a particle-count threshold."""
    daily_counts = ds["daily_beaching_counts"].values
    days_in_month = daily_counts.shape[0]
    thresholded_daily = np.where(daily_counts >= threshold, 1, 0)
    days_with_beaching = np.sum(thresholded_daily, axis=0)
    frequency = days_with_beaching.astype(float) / days_in_month
    return frequency


# =============================================================================
# MAP RENDERING (shared)
# =============================================================================

def get_cmap(name):
    if name == "thermal":
        return cmocean.cm.thermal
    return plt.get_cmap(name)


def frequency_to_rgba_image(frequency, cmap, norm):
    """Convert a (lat, lon) frequency grid into an RGBA uint8 image.

    Zero / NaN cells are transparent. The image is flipped vertically so that
    north (max lat) is at the top, matching ImageOverlay's expectation.
    """
    freq = np.where(np.isfinite(frequency), frequency, 0.0)
    rgba = cmap(norm(freq))                      # (lat, lon, 4) in 0..1
    alpha = np.where(freq > 0, 0.82, 0.0)        # hide empty cells
    rgba[..., 3] = alpha
    rgba_uint8 = (rgba * 255).astype(np.uint8)
    return np.flipud(rgba_uint8)                 # north on top


def build_legend_html(cmap, title, vmin=0.0, vmax=1.0, n=64):
    """A CSS gradient legend that lives inside the map (bottom-right)."""
    stops = []
    for i in range(n + 1):
        frac = i / n
        r, g, b, _ = cmap(frac)
        stops.append(f"rgb({int(r*255)},{int(g*255)},{int(b*255)}) {frac*100:.1f}%")
    gradient = ", ".join(stops)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    tick_html = "".join(
        f'<span style="flex:1;text-align:{"left" if t==0 else "right" if t==1 else "center"};">{t:.2f}</span>'
        for t in ticks
    )
    return f"""
    <div style="position: fixed; bottom: 24px; right: 18px; z-index: 9999;
                background: rgba(255,255,255,0.94); padding: 12px 14px 10px 14px;
                border-radius: 10px; box-shadow: 0 4px 16px rgba(0,0,0,0.18);
                font-family: 'Inter', sans-serif; width: 230px;">
        <div style="font-size: 12px; font-weight: 600; color: #0f3d4d; margin-bottom: 8px;">{title}</div>
        <div style="height: 12px; border-radius: 6px; background: linear-gradient(to right, {gradient});
                    border: 1px solid rgba(0,0,0,0.08);"></div>
        <div style="display: flex; justify-content: space-between; margin-top: 5px;
                    font-size: 10.5px; color: #5a727d;">{tick_html}</div>
    </div>
    """


def create_frequency_map(frequency, lats, lons, title, legend_title, style_name):
    """Render a frequency grid as a smooth raster overlay with an in-map legend."""
    style = MAP_STYLES[style_name]
    cmap = get_cmap(style["cmap"])
    norm = BoundaryNorm(np.arange(0.0, 1.0 + 0.025, 0.025), cmap.N)

    # Coordinates are cell centres; expand by half a cell to get the grid edges.
    dlat = float(lats[1] - lats[0]) if len(lats) > 1 else 0.05
    dlon = float(lons[1] - lons[0]) if len(lons) > 1 else 0.05
    south, north = float(lats.min()) - dlat / 2, float(lats.max()) + dlat / 2
    west, east = float(lons.min()) - dlon / 2, float(lons.max()) + dlon / 2

    m = folium.Map(
        location=[(south + north) / 2, (west + east) / 2],
        zoom_start=6,
        tiles=style["tiles"],
        attr=style["attr"],
        control_scale=True,
    )

    img = frequency_to_rgba_image(frequency, cmap, norm)
    # mercator_project=True reprojects the lat/lon raster into Web Mercator so it
    # aligns with the Leaflet basemap instead of drifting north onto land.
    ImageOverlay(
        image=img,
        bounds=[[south, west], [north, east]],
        mercator_project=True,
        opacity=1.0,
        interactive=False,
        cross_origin=False,
        zindex=1,
    ).add_to(m)

    # Title card (top-left)
    m.get_root().html.add_child(folium.Element(
        f"""<div style="position: fixed; top: 14px; left: 60px; z-index: 9999;
                background: rgba(255,255,255,0.94); padding: 10px 14px; border-radius: 10px;
                box-shadow: 0 4px 16px rgba(0,0,0,0.16); font-family:'Inter',sans-serif;">
            <div style="font-size:13px; font-weight:600; color:#0f3d4d;">{title}</div>
        </div>"""
    ))
    # Integrated legend (bottom-right)
    m.get_root().html.add_child(folium.Element(build_legend_html(cmap, legend_title)))

    # Click-to-inspect: embed non-zero cell values and a single click handler that
    # looks up the cell under the cursor and shows its frequency in a popup.
    add_click_inspect(m, frequency, lats, lons, dlat, dlon, legend_title)

    # Marine Protected Areas overlay (toggle in the in-map layer control; off by default).
    mpa = load_mpa_geojson()
    if mpa:
        add_mpa_layer(m, mpa, style["mpa_stroke"], style["mpa_fill"])
    return m


def add_mpa_layer(m, geojson_str, stroke, fill):
    """Add the WDPA Marine Protected Areas as a toggleable vector overlay.

    The layer is registered in a Leaflet layer control but NOT added to the map,
    so it starts hidden. Polygon clicks stop propagation so they take priority
    over the frequency-cell handler (the protected-area info wins on overlap)."""
    map_name = m.get_name()
    js = f"""
    <script>
    (function() {{
        function addMPA() {{
            if (typeof {map_name} === 'undefined' || typeof L === 'undefined') {{ setTimeout(addMPA, 200); return; }}
            var data = {geojson_str};
            function popupHtml(p) {{
                var nameHtml = p.url
                    ? '<a href="' + p.url + '" target="_blank" rel="noopener" style="color:#1f6f8b;text-decoration:underline;font-weight:600;">' + p.name + '</a>'
                    : '<span style="font-weight:600;color:#0f3d4d;">' + p.name + '</span>';
                var desig = p.desig + (p.desig_type && p.desig_type !== 'Not Reported' ? ' (' + p.desig_type + ')' : '');
                return '<div style="font-family:Inter,sans-serif;font-size:12.5px;color:#1a2b34;max-width:260px;line-height:1.5;">'
                     + '<div style="font-size:13.5px;margin-bottom:5px;">' + nameHtml + '</div>'
                     + '<div style="color:#5a727d;"><b style="color:#0f3d4d;">Designation:</b> ' + desig + '</div>'
                     + '<div style="color:#5a727d;"><b style="color:#0f3d4d;">IUCN category:</b> ' + p.iucn + '</div></div>';
            }}
            var mpaLayer = L.geoJSON(data, {{
                style: function() {{ return {{color:'{stroke}', weight:1.4, fillColor:'{fill}', fillOpacity:0.12, opacity:0.9}}; }},
                onEachFeature: function(feature, lyr) {{
                    lyr.on('click', function(e) {{
                        L.DomEvent.stopPropagation(e);
                        L.popup({{maxWidth:280}}).setLatLng(e.latlng).setContent(popupHtml(feature.properties)).openOn({map_name});
                    }});
                    lyr.on('mouseover', function() {{ lyr.setStyle({{fillOpacity:0.32, weight:2.4}}); }});
                    lyr.on('mouseout', function() {{ mpaLayer.resetStyle(lyr); }});
                }}
            }});
            L.control.layers(null, {{'Marine Protected Areas': mpaLayer}}, {{collapsed:false, position:'topright'}}).addTo({map_name});
        }}
        addMPA();
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(js))


def add_click_inspect(m, frequency, lats, lons, dlat, dlon, value_label):
    """Attach a Leaflet click handler that reports the grid frequency at the
    clicked location (replaces per-cell popups, which a raster overlay lacks)."""
    freq = np.where(np.isfinite(frequency), frequency, 0.0)
    ii, jj = np.where(freq > 0)
    vals = np.round(freq[ii, jj].astype(float), 3)
    sparse = {f"{int(a)}_{int(b)}": float(c) for a, b, c in zip(ii, jj, vals)}

    map_name = m.get_name()
    js = f"""
    <script>
    (function() {{
        var freqData = {json.dumps(sparse)};
        var lat0 = {float(lats[0])}, lon0 = {float(lons[0])};
        var dlat = {dlat}, dlon = {dlon};
        var nlat = {len(lats)}, nlon = {len(lons)};
        function bindInspect() {{
            if (typeof {map_name} === 'undefined') {{ setTimeout(bindInspect, 200); return; }}
            {map_name}.on('click', function(e) {{
                var i = Math.round((e.latlng.lat - lat0) / dlat);
                var j = Math.round((e.latlng.lng - lon0) / dlon);
                if (i < 0 || i >= nlat || j < 0 || j >= nlon) return;
                var v = freqData[i + '_' + j];
                var html;
                if (v === undefined) {{
                    html = '<div style="font-family:Inter,sans-serif;font-size:12px;color:#5a727d;">'
                         + 'No data at this cell</div>';
                }} else {{
                    html = '<div style="font-family:Inter,sans-serif;font-size:12.5px;color:#1a2b34;">'
                         + '<b style="color:#0f3d4d;">{value_label}</b><br>'
                         + '<span style="font-size:18px;font-weight:700;color:#1f6f8b;">' + v.toFixed(3) + '</span><br>'
                         + '<span style="color:#5a727d;">Lat ' + e.latlng.lat.toFixed(2)
                         + ' &middot; Lon ' + e.latlng.lng.toFixed(2) + '</span></div>';
                }}
                L.popup({{maxWidth: 220}}).setLatLng(e.latlng).setContent(html).openOn({map_name});
            }});
        }}
        bindInspect();
    }})();
    </script>
    """
    m.get_root().html.add_child(folium.Element(js))


# =============================================================================
# VIEWS
# =============================================================================

def view_beached_thresholds(months, theme):
    st.subheader("Threshold testing")
    st.markdown(
        '<p class="muted">Test how different particle-count thresholds reshape the '
        'active beaching footprint across all twelve months.</p>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        threshold_input = st.text_input(
            "Thresholds (comma-separated)",
            value="1, 10, 50, 100, 500",
            help="Minimum beached-particle counts to keep a cell.",
            key="bt_thresholds",
        )
    with col2:
        st.write("")
        st.write("")
        run = st.button("Generate analysis", use_container_width=True, key="bt_run")

    if not run:
        return

    try:
        thresholds = sorted({int(x.strip()) for x in threshold_input.split(",") if x.strip()})
    except ValueError:
        st.error("Please enter whole numbers separated by commas.")
        return
    if not thresholds:
        st.error("Enter at least one threshold value.")
        return

    progress = st.progress(0.0)
    for idx, (month_num, fp) in enumerate(sorted(months.items())):
        month_name = MONTH_INFO[month_num]["name"]
        progress.progress((idx + 1) / len(months))
        ds = load_dataset(fp)
        total_particles = ds["total_particle_counts"].values.flatten()
        active_particles = total_particles[total_particles > 0]
        if len(active_particles) == 0:
            continue

        stats, active_loss_stats, active_baseline = calculate_threshold_statistics(total_particles, thresholds)

        with st.expander(f"{month_name} 2024", expanded=(idx == 0)):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total beached", f"{ds.attrs.get('total_beached_particles', 0):,}")
            c2.metric("Active cells", f"{active_baseline:,}")
            c3.metric("Max / cell", f"{int(total_particles.max()):,}")
            c4.metric("Mean (active)", f"{active_particles.mean():.1f}")

            p1, p2 = st.columns(2)
            with p1:
                st.pyplot(create_distribution_histogram(active_particles, thresholds, month_name, active_baseline, theme))
            with p2:
                st.pyplot(create_threshold_impact_plot(active_loss_stats, thresholds, month_name, active_baseline, theme))

            table = [{
                "Threshold": f"≥ {t}",
                "Cells kept": stats[t]["active_cells"],
                "% kept": f"{active_loss_stats[t]['percentage_of_active_kept']:.1f}%",
                "Cells lost": active_loss_stats[t]["cells_lost_from_active"],
                "% lost": f"{active_loss_stats[t]['percentage_of_active_lost']:.1f}%",
            } for t in thresholds]
            render_table(pd.DataFrame(table), theme)

    progress.empty()
    st.success("Analysis complete.")


def view_beached_maps(months):
    st.subheader("Frequency maps")
    st.markdown(
        '<p class="muted">Apply a particle-count threshold and explore the resulting '
        'beaching-frequency field interactively.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        threshold = st.number_input("Threshold", min_value=1, max_value=10000, value=50, step=10, key="bm_threshold")
    with c2:
        month_num = st.selectbox("Month", options=sorted(months.keys()),
                                 format_func=lambda x: MONTH_INFO[x]["name"], key="bm_month")
    with c3:
        style_name = st.selectbox("Map style", options=list(MAP_STYLES.keys()), key="bm_style")

    if not st.button("Generate map", use_container_width=True, key="bm_generate"):
        return

    month_name = MONTH_INFO[month_num]["name"]
    ds = load_dataset(months[month_num])
    frequency = apply_threshold_to_frequency(ds, threshold)
    lats, lons = ds["lat"].values, ds["lon"].values

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Threshold", f"≥ {threshold}")
    c2.metric("Active cells", f"{int(np.sum(frequency > 0)):,}")
    c3.metric("Max frequency", f"{frequency.max():.3f}")
    c4.metric("Total beached", f"{ds.attrs.get('total_beached_particles', 0):,}")

    m = create_frequency_map(
        frequency, lats, lons,
        title=f"Beaching frequency — {month_name} 2024  ·  ≥ {threshold} particles",
        legend_title="Beaching occurrence frequency",
        style_name=style_name,
    )
    st.components.v1.html(m._repr_html_(), height=620)


def view_surface_maps(months):
    st.subheader("Frequency maps")
    st.markdown(
        '<p class="muted">Explore monthly surface-particle occurrence frequency across '
        'the basin.</p>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        month_num = st.selectbox("Month", options=sorted(months.keys()),
                                 format_func=lambda x: MONTH_INFO[x]["name"], key="sm_month")
    with c2:
        style_name = st.selectbox("Map style", options=list(MAP_STYLES.keys()), key="sm_style")

    if not st.button("Generate map", use_container_width=True, key="sm_generate"):
        return

    month_name = MONTH_INFO[month_num]["name"]
    ds = load_dataset(months[month_num])
    frequency = np.squeeze(ds["frequency"].values)
    lats, lons = ds["lat"].values, ds["lon"].values

    c1, c2, c3 = st.columns(3)
    c1.metric("Active cells", f"{int(np.sum(frequency > 0)):,}")
    c2.metric("Max frequency", f"{np.nanmax(frequency):.3f}")
    c3.metric("Mean (active)", f"{frequency[frequency > 0].mean():.3f}")

    m = create_frequency_map(
        frequency, lats, lons,
        title=f"Surface particle frequency — {month_name} 2024",
        legend_title="Surface occurrence frequency",
        style_name=style_name,
    )
    st.components.v1.html(m._repr_html_(), height=620)


# =============================================================================
# APP
# =============================================================================

def main():
    inject_analytics(GA_MEASUREMENT_ID)

    # Header row: theme toggle (left) and partner logos (right). Dark is the
    # default; the switch turns Light mode on. Read the toggle before injecting
    # CSS so the page renders in the chosen theme on the same run.
    toggle_col, logo_col = st.columns([1, 2])
    with toggle_col:
        light = st.toggle("Light mode", value=st.session_state.get("light_mode", False), key="light_toggle")
    st.session_state["light_mode"] = light
    theme = THEMES["light"] if light else THEMES["dark"]

    inject_css(theme)

    with logo_col:
        render_logos(theme)

    beached_months = load_beached_months()
    surface_months = load_surface_months()

    hero(
        "Particle Analysis Dashboard",
        "Lagrangian beaching and surface particle dynamics across the basin · 2024",
    )

    beached_tab, surface_tab = st.tabs(["Beached particles", "Surface particles"])

    with beached_tab:
        if not beached_months:
            st.error(f"No beached data files found in '{BEACHED_DIR}'.")
        else:
            thr_tab, map_tab = st.tabs(["Threshold testing", "Frequency maps"])
            with thr_tab:
                view_beached_thresholds(beached_months, theme)
            with map_tab:
                view_beached_maps(beached_months)

    with surface_tab:
        if not surface_months:
            st.error(f"No surface data files found in '{SURFACE_DIR}'.")
        else:
            view_surface_maps(surface_months)

    render_footer(theme)


if __name__ == "__main__":
    main()
