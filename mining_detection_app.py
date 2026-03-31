"""
╔══════════════════════════════════════════════════════════════════╗
║   SENTINEL WATCH — Open Source Illegal Mining Detection          ║
║   Built on Sentinel-2 + Microsoft Planetary Computer             ║
║   pip install streamlit pystac-client planetary-computer         ║
║        stackstac scikit-learn requests matplotlib numpy          ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap
import requests
import warnings
import io
import base64
from datetime import datetime

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentinel Watch — Illegal Mining Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── GLOBAL DARK THEME CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

  :root {
    --bg:        #080c14;
    --bg2:       #0d1525;
    --bg3:       #111a2e;
    --border:    #1e3058;
    --accent:    #00d4ff;
    --accent2:   #ff4444;
    --accent3:   #00ff88;
    --accent4:   #ffaa00;
    --text:      #e8f0fe;
    --muted:     #7a8fb0;
    --card-bg:   #0d1525;
  }

  /* Override Streamlit root backgrounds */
  .stApp, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
  }
  [data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border);
  }
  section[data-testid="stSidebarContent"] {
    background: var(--bg2) !important;
  }
  [data-testid="stHeader"] {
    background: transparent !important;
  }

  /* Typography */
  html, body, .stApp {
    font-family: 'Syne', sans-serif !important;
    color: var(--text) !important;
  }
  code, pre, .stCodeBlock {
    font-family: 'Space Mono', monospace !important;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 1rem !important;
  }
  [data-testid="metric-container"] label {
    color: var(--muted) !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-family: 'Space Mono', monospace !important;
  }
  [data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-size: 1.8rem !important;
    font-weight: 800 !important;
    font-family: 'Syne', sans-serif !important;
  }

  /* Tabs */
  [data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
  }
  [data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--muted) !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 1.2rem !important;
    text-transform: uppercase !important;
  }
  [data-testid="stTabs"] [aria-selected="true"] {
    color: var(--accent) !important;
    border-bottom: 2px solid var(--accent) !important;
  }
  [data-testid="stTabsContent"] {
    background: var(--bg) !important;
    padding-top: 1.5rem !important;
  }

  /* Buttons */
  .stButton > button {
    background: linear-gradient(135deg, #0a2a6e 0%, #0d1f4a 100%) !important;
    color: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    padding: 0.6rem 1.5rem !important;
    transition: all 0.2s ease !important;
  }
  .stButton > button:hover {
    background: var(--accent) !important;
    color: var(--bg) !important;
    box-shadow: 0 0 20px rgba(0,212,255,0.4) !important;
  }

  /* Run button special */
  .run-btn > button {
    background: linear-gradient(135deg, #00d4ff22, #0090aa33) !important;
    border: 2px solid var(--accent) !important;
    font-size: 1rem !important;
    padding: 0.8rem 2rem !important;
    width: 100% !important;
  }

  /* Inputs */
  .stNumberInput input, .stTextInput input, .stSelectbox select {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.85rem !important;
  }

  /* Sliders */
  [data-testid="stSlider"] [data-baseweb="slider"] {
    color: var(--accent) !important;
  }

  /* Expander */
  [data-testid="stExpander"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
  }
  [data-testid="stExpander"] summary {
    color: var(--accent) !important;
    font-family: 'Space Mono', monospace !important;
  }

  /* Custom alert boxes */
  .alert-critical {
    background: rgba(255,68,68,0.1);
    border: 1px solid #ff4444;
    border-left: 4px solid #ff4444;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #ff8888;
  }
  .alert-warning {
    background: rgba(255,170,0,0.1);
    border: 1px solid #ffaa00;
    border-left: 4px solid #ffaa00;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #ffcc55;
  }
  .alert-ok {
    background: rgba(0,255,136,0.08);
    border: 1px solid #00ff88;
    border-left: 4px solid #00ff88;
    border-radius: 4px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: #55ffaa;
  }

  /* Section headers */
  .section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 800;
    color: var(--text);
    border-bottom: 2px solid var(--border);
    padding-bottom: 0.5rem;
    margin-bottom: 1.5rem;
    letter-spacing: 0.02em;
  }
  .section-header span {
    color: var(--accent);
  }

  /* Index cards */
  .idx-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
  }
  .idx-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.25rem;
  }
  .idx-name {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent);
  }
  .idx-desc {
    font-size: 0.82rem;
    color: var(--muted);
    margin-top: 0.3rem;
    line-height: 1.5;
  }

  /* Progress bar override */
  [data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, var(--accent), var(--accent3)) !important;
  }

  /* Selectbox */
  [data-testid="stSelectbox"] > div > div {
    background: var(--bg3) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
  }

  /* Divider */
  hr {
    border-color: var(--border) !important;
    margin: 1.5rem 0 !important;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: var(--bg2); }
  ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--accent); }
</style>
""", unsafe_allow_html=True)

# ── HERO HEADER ───────────────────────────────────────────────────────────
st.markdown("""
<div style="
  background: linear-gradient(135deg, #080c14 0%, #0a1628 40%, #0d1f4a 100%);
  border: 1px solid #1e3058;
  border-radius: 12px;
  padding: 2.5rem 2rem 2rem 2rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
">
  <div style="
    position: absolute; top: 0; right: 0; width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,212,255,0.06) 0%, transparent 70%);
    pointer-events: none;
  "></div>
  <div style="
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.25em;
    color: #00d4ff;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
  ">🛰️ Open Source · Sentinel-2 L2A · Planetary Computer</div>
  <div style="
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    color: #e8f0fe;
    line-height: 1.1;
    margin-bottom: 0.6rem;
  ">SENTINEL<span style='color:#00d4ff;'>WATCH</span></div>
  <div style="
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    color: #7a8fb0;
    max-width: 600px;
    line-height: 1.6;
  ">
    Detect illegal mining from space using multi-spectral satellite imagery.
    Free, open-source, and requires no satellite expertise. 
    Powered by <strong style='color:#00d4ff;'>ESA Sentinel-2</strong> via Microsoft Planetary Computer.
  </div>
  <div style="
    display: flex; gap: 1rem; margin-top: 1.2rem; flex-wrap: wrap;
  ">
    <span style="background:#0d2244;border:1px solid #1e4080;border-radius:20px;padding:0.25rem 0.8rem;font-family:'Space Mono',monospace;font-size:0.72rem;color:#00d4ff;">NDVI · BSI · NDWI · NBR</span>
    <span style="background:#0d2244;border:1px solid #1e4080;border-radius:20px;padding:0.25rem 0.8rem;font-family:'Space Mono',monospace;font-size:0.72rem;color:#00ff88;">K-Means Classification</span>
    <span style="background:#0d2244;border:1px solid #1e4080;border-radius:20px;padding:0.25rem 0.8rem;font-family:'Space Mono',monospace;font-size:0.72rem;color:#ffaa00;">Temporal Anomaly Detection</span>
    <span style="background:#0d2244;border:1px solid #1e4080;border-radius:20px;padding:0.25rem 0.8rem;font-family:'Space Mono',monospace;font-size:0.72rem;color:#ff6688;">False Positive Filtering</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR — CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:800;
                color:#00d4ff;margin-bottom:0.2rem;">⚙️ Configuration</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#7a8fb0;
                margin-bottom:1.2rem;">Define area of interest + parameters</div>
    """, unsafe_allow_html=True)

    # ── PRESET LOCATIONS ──
    st.markdown("**📍 Quick Presets**")
    preset = st.selectbox("Known mining regions", [
        "Custom",
        "Jharkhand, India (Coal Belt)",
        "Amazon, Brazil (Gold Mining)",
        "DRC Congo (Artisanal Mining)",
        "Madre de Dios, Peru",
        "Katanga, DRC (Copper Belt)",
    ])

    preset_coords = {
        "Jharkhand, India (Coal Belt)":     (85.8, 23.5, 86.2, 23.8),
        "Amazon, Brazil (Gold Mining)":      (-60.5, -7.5, -60.0, -7.0),
        "DRC Congo (Artisanal Mining)":      (27.5, -3.0, 28.0, -2.5),
        "Madre de Dios, Peru":               (-70.5, -12.8, -70.0, -12.3),
        "Katanga, DRC (Copper Belt)":        (27.4, -11.7, 27.9, -11.2),
    }

    if preset != "Custom" and preset in preset_coords:
        def_lon_min, def_lat_min, def_lon_max, def_lat_max = preset_coords[preset]
    else:
        def_lon_min, def_lat_min, def_lon_max, def_lat_max = 85.8, 23.5, 86.2, 23.8

    st.markdown("**🗺️ Area of Interest (Bounding Box)**")
    col1, col2 = st.columns(2)
    with col1:
        lon_min = st.number_input("Lon Min", value=float(def_lon_min), format="%.4f", step=0.1)
        lat_min = st.number_input("Lat Min", value=float(def_lat_min), format="%.4f", step=0.1)
    with col2:
        lon_max = st.number_input("Lon Max", value=float(def_lon_max), format="%.4f", step=0.1)
        lat_max = st.number_input("Lat Max", value=float(def_lat_max), format="%.4f", step=0.1)

    st.markdown("**📅 Temporal Range**")
    year_before = st.selectbox("Baseline Year (Before)", list(range(2017, 2024)), index=2)
    year_after  = st.selectbox("Analysis Year (After)",  list(range(2017, 2025)), index=6)
    years_temporal = st.multiselect(
        "Temporal Analysis Years",
        options=list(range(2017, 2025)),
        default=[2019, 2020, 2021, 2022, 2023],
    )

    st.markdown("**🔧 Analysis Parameters**")
    cloud_cover   = st.slider("Max Cloud Cover (%)", 5, 50, 20, 5)
    mining_thresh = st.slider("Mining Score Threshold", 0.2, 0.8, 0.5, 0.05)
    resolution    = st.selectbox("Resolution (m)", [20, 60], index=1)
    n_clusters    = st.slider("K-Means Clusters", 3, 6, 4)

    st.markdown("**⚖️ Index Weights**")
    with st.expander("Customize fusion weights"):
        w_ndvi = st.slider("NDVI weight (vegetation loss)", 0.0, 1.0, 0.35, 0.05)
        w_bsi  = st.slider("BSI weight (bare soil gain)",   0.0, 1.0, 0.30, 0.05)
        w_ndwi = st.slider("NDWI weight (water turbidity)", 0.0, 1.0, 0.20, 0.05)
        w_nbr  = st.slider("NBR weight (land disturbance)", 0.0, 1.0, 0.15, 0.05)
        total  = w_ndvi + w_bsi + w_ndwi + w_nbr
        if abs(total - 1.0) > 0.01:
            st.warning(f"⚠️ Weights sum to {total:.2f} (should be 1.0)")

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.68rem;color:#3a5080;
                line-height:1.6;text-align:center;">
    Data: ESA Sentinel-2 L2A<br>
    Source: Microsoft Planetary Computer<br>
    Resolution: 60m per pixel<br>
    License: Open-source (MIT)
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# HELPER: BUILD AOI
# ══════════════════════════════════════════════════════════════════════════
aoi = {
    "type": "Polygon",
    "coordinates": [[
        [lon_min, lat_min],
        [lon_max, lat_min],
        [lon_max, lat_max],
        [lon_min, lat_max],
        [lon_min, lat_min],
    ]]
}

# ══════════════════════════════════════════════════════════════════════════
# MATPLOTLIB DARK THEME
# ══════════════════════════════════════════════════════════════════════════
BG      = "#080c14"
BG2     = "#0d1525"
BORDER  = "#1e3058"
ACCENT  = "#00d4ff"
RED     = "#ff4444"
GREEN   = "#00ff88"
ORANGE  = "#ffaa00"

def dark_fig(nrows=1, ncols=1, figsize=(10, 6), **kw):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, **kw)
    fig.patch.set_facecolor(BG)
    for ax in (np.array(axes).flatten() if hasattr(axes, '__iter__') else [axes]):
        ax.set_facecolor(BG2)
        ax.tick_params(colors="#7a8fb0")
        ax.spines[:].set_color(BORDER)
        ax.xaxis.label.set_color("#7a8fb0")
        ax.yaxis.label.set_color("#7a8fb0")
        ax.title.set_color("#e8f0fe")
    return fig, axes

def fig_to_st(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)

def fig_to_download(fig, fname):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight", facecolor=BG)
    buf.seek(0)
    return buf.getvalue(), fname

# ══════════════════════════════════════════════════════════════════════════
# CORE ANALYSIS FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_catalog():
    import pystac_client
    import planetary_computer
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

@st.cache_data(show_spinner=False)
def fetch_stack(year, _aoi, cloud, res, epsg=32645):
    import stackstac
    catalog = get_catalog()
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=_aoi,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": cloud}},
    )
    items = list(search.items())
    if not items:
        return None, 0
    stack = stackstac.stack(
        items[:1],
        assets=["B04", "B08", "B11", "B03"],
        resolution=res,
        epsg=epsg,
        chunksize=512,
    ).compute()
    return stack, len(items)

def compute_indices(stack):
    red   = stack.sel(band="B04").values[0].astype(float)
    nir   = stack.sel(band="B08").values[0].astype(float)
    swir  = stack.sel(band="B11").values[0].astype(float)
    green = stack.sel(band="B03").values[0].astype(float)
    for arr in [red, nir, swir, green]:
        arr[arr == 0] = np.nan
    ndvi = (nir - red)    / (nir + red    + 1e-10)
    bsi  = ((swir + red)  - (nir + green)) / ((swir + red) + (nir + green) + 1e-10)
    ndwi = (green - nir)  / (green + nir   + 1e-10)
    nbr  = (nir - swir)   / (nir + swir    + 1e-10)
    return ndvi, bsi, ndwi, nbr

def normalise(arr):
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, 0, None)
    return arr / arr.max() if arr.max() > 0 else arr

def compute_score(ndvi_b, bsi_b, ndwi_b, nbr_b,
                  ndvi_a, bsi_a, ndwi_a, nbr_a,
                  w_ndvi=0.35, w_bsi=0.30, w_ndwi=0.20, w_nbr=0.15):
    d_ndvi = normalise(ndvi_b - ndvi_a)
    d_bsi  = normalise(bsi_a  - bsi_b)
    d_ndwi = normalise(ndwi_b - ndwi_a)
    d_nbr  = normalise(nbr_b  - nbr_a)
    score  = w_ndvi*d_ndvi + w_bsi*d_bsi + w_ndwi*d_ndwi + w_nbr*d_nbr
    return score, d_ndvi, d_bsi, d_ndwi, d_nbr

# ══════════════════════════════════════════════════════════════════════════
# RUN ANALYSIS BUTTON
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="run-btn">', unsafe_allow_html=True)
run = st.button("🛰️  RUN ANALYSIS — Fetch Satellite Data & Detect Mining")
st.markdown("</div>", unsafe_allow_html=True)

if not run:
    # ── LANDING INFO ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="margin-top:2rem;">
    <div class="section-header">How <span>SentinelWatch</span> Works</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    info = [
        ("01", "NDVI", "Normalized Difference Vegetation Index",
         "Measures live green vegetation. Mining causes sharp drops — forests become bare ground.",
         "#00ff88"),
        ("02", "BSI",  "Bare Soil Index",
         "Detects exposed earth. High BSI in previously green areas indicates excavation activity.",
         "#ffaa00"),
        ("03", "NDWI", "Normalized Difference Water Index",
         "Tracks water turbidity. Mining runoff causes rivers and ponds to turn murky.",
         "#00d4ff"),
        ("04", "NBR",  "Normalized Burn Ratio",
         "Originally for fire detection — also captures severe land surface disturbance.",
         "#ff6688"),
    ]
    for col, (num, abbr, full, desc, color) in zip([c1,c2,c3,c4], info):
        with col:
            st.markdown(f"""
            <div class="idx-card">
              <div class="idx-title">Index {num}</div>
              <div class="idx-name" style="color:{color};">{abbr}</div>
              <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                          color:#3a5080;margin-bottom:0.4rem;">{full}</div>
              <div class="idx-desc">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:1.5rem;">
    <div class="section-header">Analysis <span>Pipeline</span></div>
    </div>
    """, unsafe_allow_html=True)

    steps = [
        ("🛰️", "Fetch",      "Download Sentinel-2 imagery from Microsoft Planetary Computer for baseline and analysis years"),
        ("📐", "Compute",    "Calculate NDVI, BSI, NDWI, NBR for both time periods at 60m resolution"),
        ("📊", "Delta",      "Subtract before/after for each index to isolate change signals"),
        ("🔥", "Fuse",       "Weighted combination of all 4 delta indices into single mining probability score"),
        ("🗺️", "Classify",   "K-Means clustering to map forest, water, degraded land, and active mining zones"),
        ("📈", "Temporal",   "Multi-year anomaly detection with σ-based alert thresholds"),
        ("🚨", "Filter",     "Remove false positives using OpenStreetMap land-use data via Overpass API"),
        ("📋", "Report",     "Generate downloadable summary with area estimates, scores, and risk classification"),
    ]
    for i, (icon, title, desc) in enumerate(steps):
        st.markdown(f"""
        <div style="display:flex;align-items:flex-start;gap:1rem;padding:0.7rem 0;
                    border-bottom:1px solid {BORDER};">
          <div style="font-size:1.3rem;width:2rem;flex-shrink:0;">{icon}</div>
          <div>
            <span style="font-family:'Space Mono',monospace;font-size:0.75rem;
                         color:#00d4ff;text-transform:uppercase;letter-spacing:0.1em;">{title}</span>
            <span style="font-size:0.88rem;color:#7a8fb0;margin-left:1rem;">{desc}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2rem;background:#0d1525;border:1px solid #1e3058;
                border-radius:8px;padding:1.2rem;font-family:'Space Mono',monospace;
                font-size:0.78rem;color:#7a8fb0;line-height:1.8;">
    <strong style="color:#00d4ff;">ℹ️  Getting Started</strong><br>
    1. Select a preset region or enter custom coordinates in the sidebar<br>
    2. Choose your baseline year (before mining) and analysis year (after)<br>
    3. Click <strong style="color:#e8f0fe;">RUN ANALYSIS</strong> above<br>
    4. All results download automatically — share with journalists, NGOs, or regulators
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS EXECUTION
# ══════════════════════════════════════════════════════════════════════════
progress_bar = st.progress(0)
status_text  = st.empty()

def update(pct, msg):
    progress_bar.progress(pct)
    status_text.markdown(f"""
    <div style="font-family:'Space Mono',monospace;font-size:0.8rem;
                color:#00d4ff;padding:0.4rem 0;">⟳ {msg}</div>
    """, unsafe_allow_html=True)

# ── STEP 1: FETCH DATA ────────────────────────────────────────────────────
update(5, f"Connecting to Planetary Computer...")

try:
    catalog = get_catalog()
except Exception as e:
    st.error(f"❌ Could not connect to Planetary Computer: {e}")
    st.stop()

update(10, f"Fetching {year_before} baseline imagery (cloud < {cloud_cover}%)...")
stack_b, n_before = fetch_stack(year_before, aoi, cloud_cover, resolution)

if stack_b is None:
    st.error(f"❌ No Sentinel-2 images found for {year_before} with cloud cover < {cloud_cover}%. Try increasing the cloud cover threshold.")
    st.stop()

update(30, f"Fetching {year_after} analysis imagery...")
stack_a, n_after = fetch_stack(year_after, aoi, cloud_cover, resolution)

if stack_a is None:
    st.error(f"❌ No Sentinel-2 images found for {year_after}. Try a different year or looser cloud cover.")
    st.stop()

update(50, "Computing spectral indices...")
ndvi_b, bsi_b, ndwi_b, nbr_b = compute_indices(stack_b)
ndvi_a, bsi_a, ndwi_a, nbr_a = compute_indices(stack_a)

update(60, "Fusing indices into mining probability score...")
mining_score, d_ndvi, d_bsi, d_ndwi, d_nbr = compute_score(
    ndvi_b, bsi_b, ndwi_b, nbr_b,
    ndvi_a, bsi_a, ndwi_a, nbr_a,
    w_ndvi, w_bsi, w_ndwi, w_nbr,
)
mining_mask   = mining_score > mining_thresh
px_area_ha    = (resolution / 100) ** 2 * 100   # hectares per pixel
affected_ha   = mining_mask.sum() * px_area_ha
critical_ha   = (mining_score > 0.65).sum() * px_area_ha
peak_score    = float(mining_score.max())

update(70, "Running K-Means land cover classification...")

# ── STEP 2: K-MEANS ───────────────────────────────────────────────────────
from sklearn.cluster import KMeans

h, w = ndvi_a.shape
feats_a = np.stack([np.nan_to_num(ndvi_a), np.nan_to_num(bsi_a),
                    np.nan_to_num(ndwi_a), np.nan_to_num(nbr_a)], axis=-1).reshape(-1, 4)
feats_b = np.stack([np.nan_to_num(ndvi_b), np.nan_to_num(bsi_b),
                    np.nan_to_num(ndwi_b), np.nan_to_num(nbr_b)], axis=-1).reshape(-1, 4)

kmeans  = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans.fit(feats_a)
labels_a = kmeans.predict(feats_a).reshape(h, w)
labels_b = kmeans.predict(feats_b).reshape(h, w)

centers = kmeans.cluster_centers_
forest_cls   = int(np.argmax(centers[:, 0]))
mining_cls   = int(np.argmax(centers[:, 1]))
water_cls    = int(np.argmax(centers[:, 2]))
others       = list(set(range(n_clusters)) - {forest_cls, mining_cls, water_cls})

cluster_names  = {forest_cls: ("Forest/Vegetation", "#2d8a2d")}
cluster_names[water_cls]  = ("Water Bodies", "#3399ff")
cluster_names[mining_cls] = ("Active Mining/Bare Soil", "#ff2222")
for i, c in enumerate(others):
    cluster_names[c] = (f"Degraded Land {i+1}", "#ff8800" if i == 0 else "#cc6600")

update(80, "Fetching OpenStreetMap data for false-positive filtering...")

# ── STEP 3: FALSE POSITIVE FILTER ────────────────────────────────────────
fp_mask = np.zeros((h, w), dtype=bool)
try:
    overpass_url = "http://overpass-api.de/api/interpreter"
    q = f"""
[out:json][timeout:25];
(
  way["landuse"~"residential|farmland|farm|industrial"]({lat_min},{lon_min},{lat_max},{lon_max});
  way["place"~"village|town|city|hamlet"]({lat_min},{lon_min},{lat_max},{lon_max});
);
out geom;
"""
    resp = requests.get(overpass_url, params={"data": q}, timeout=30)
    osm_data = resp.json().get("elements", [])
    for el in osm_data:
        if "geometry" not in el:
            continue
        for nd in el["geometry"]:
            col_ = int((nd["lon"] - lon_min) / (lon_max - lon_min) * w)
            row_ = int((lat_max - nd["lat"]) / (lat_max - lat_min) * h)
            if 0 <= row_ < h and 0 <= col_ < w:
                fp_mask[row_, col_] = True
    osm_ok = True
except Exception:
    osm_ok = False

mining_score_filtered = mining_score.copy()
mining_score_filtered[fp_mask] = 0
filtered_ha  = (mining_score_filtered > mining_thresh).sum() * px_area_ha
removed_ha   = affected_ha - filtered_ha

update(90, "Running temporal anomaly detection...")

# ── STEP 4: TEMPORAL ANALYSIS ─────────────────────────────────────────────
temporal_data = {}
for yr in sorted(years_temporal):
    stk, _ = fetch_stack(yr, aoi, cloud_cover, resolution)
    if stk is not None:
        ni, bi, wi, nbi = compute_indices(stk)
        sm = (w_ndvi*normalise(ni) + w_bsi*normalise(bi) +
              w_ndwi*normalise(wi) + w_nbr*normalise(nbi))
        temporal_data[yr] = {
            "score": float(np.nanmean(sm)),
            "ndvi":  float(np.nanmean(ni)),
            "ndvi_map": ni,
        }

if temporal_data:
    t_years  = sorted(temporal_data.keys())
    t_scores = np.array([temporal_data[y]["score"] for y in t_years])
    t_mean   = float(np.mean(t_scores))
    t_std    = float(np.std(t_scores))
    t_thresh = t_mean + 1.0 * t_std

update(100, "Analysis complete!")
progress_bar.empty()
status_text.empty()

# ══════════════════════════════════════════════════════════════════════════
# SUMMARY METRICS BANNER
# ══════════════════════════════════════════════════════════════════════════
risk_level = "🔴 CRITICAL" if peak_score > 0.65 else ("🟡 SUSPECTED" if peak_score > 0.4 else "🟢 LOW RISK")
risk_color = RED if peak_score > 0.65 else (ORANGE if peak_score > 0.4 else GREEN)

st.markdown(f"""
<div style="background:linear-gradient(135deg,{BG2},{BG});border:1px solid {risk_color};
            border-radius:10px;padding:1.2rem 1.5rem;margin-bottom:1.5rem;
            box-shadow: 0 0 30px {risk_color}22;">
  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">
    <div style="font-family:'Syne',sans-serif;font-size:1.6rem;font-weight:800;
                color:{risk_color};">{risk_level}</div>
    <div style="font-family:'Space Mono',monospace;font-size:0.75rem;color:#7a8fb0;">
      {year_before} → {year_after} · {lon_min:.2f},{lat_min:.2f} to {lon_max:.2f},{lat_max:.2f}
    </div>
  </div>
  <div style="font-family:'Space Mono',monospace;font-size:0.78rem;color:#7a8fb0;">
    Peak mining score <strong style="color:{risk_color};">{peak_score:.3f}</strong> ·
    Affected area <strong style="color:#e8f0fe;">{affected_ha:.0f} ha</strong> ·
    Critical zone <strong style="color:{RED};">{critical_ha:.0f} ha</strong> ·
    After FP filter <strong style="color:{GREEN};">{filtered_ha:.0f} ha</strong>
  </div>
</div>
""", unsafe_allow_html=True)

mc1, mc2, mc3, mc4, mc5 = st.columns(5)
mc1.metric("Peak Score",    f"{peak_score:.3f}")
mc2.metric("Affected Area", f"{affected_ha:.0f} ha")
mc3.metric("Critical Zone", f"{critical_ha:.0f} ha")
mc4.metric("FP-Filtered",   f"{filtered_ha:.0f} ha")
mc5.metric("Risk Level",    risk_level.split()[1])

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 4 INDICES",
    "🔥 FUSION MAP",
    "🗺️ LAND COVER",
    "📈 TEMPORAL",
    "🎬 ANIMATION",
    "🚨 FP FILTER",
    "🔬 PROBABILITY",
    "📋 REPORT",
])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — 4 SPECTRAL INDICES
# ══════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">4 Spectral <span>Indices</span> — Before vs After</div>', unsafe_allow_html=True)

    index_info = [
        ("NDVI", "Normalized Difference Vegetation Index",
         ndvi_b, ndvi_a, d_ndvi,
         "RdYlGn", "RdYlGn", "RdYlGn_r",
         "Vegetation loss → mining signature", "#00ff88",
         "NDVI = (NIR − RED) / (NIR + RED)",
         "NDVI drops from 0.6+ in forests to near-zero over excavated land. "
         "The delta map highlights exact zones of vegetation destruction."),
        ("BSI",  "Bare Soil Index",
         bsi_b, bsi_a, d_bsi,
         "PiYG_r", "PiYG_r", "OrRd",
         "Bare soil increase → excavation", "#ffaa00",
         "BSI = ((SWIR + RED) − (NIR + GREEN)) / ((SWIR + RED) + (NIR + GREEN))",
         "High BSI values reveal exposed mineral soil. Mining dramatically increases "
         "BSI as overburden is removed and topsoil destroyed."),
        ("NDWI", "Normalized Difference Water Index",
         ndwi_b, ndwi_a, d_ndwi,
         "RdBu", "RdBu", "RdBu",
         "Water turbidity → mining runoff", "#00d4ff",
         "NDWI = (GREEN − NIR) / (GREEN + NIR)",
         "Mining operations produce acidic, sediment-laden water. Changes in NDWI "
         "near water bodies indicate mine effluent contamination."),
        ("NBR",  "Normalized Burn Ratio",
         nbr_b, nbr_a, d_nbr,
         "RdYlBu", "RdYlBu", "YlOrRd",
         "Land disturbance → surface damage", "#ff6688",
         "NBR = (NIR − SWIR) / (NIR + SWIR)",
         "Originally designed for post-fire assessment, NBR also captures severe "
         "land surface disruption from open-cast mining operations."),
    ]

    for abbr, full, arr_b, arr_a, arr_d, cmap_b, cmap_a, cmap_d, meaning, color, formula, explain in index_info:
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:8px;
                    padding:1rem 1.2rem;margin-bottom:0.5rem;">
          <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.3rem;">
            <span style="font-family:'Syne',sans-serif;font-size:1.3rem;font-weight:800;
                         color:{color};">{abbr}</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.72rem;color:#7a8fb0;">{full}</span>
            <span style="font-family:'Space Mono',monospace;font-size:0.68rem;
                         background:#111a2e;border:1px solid {BORDER};border-radius:3px;
                         padding:0.1rem 0.5rem;color:#aabbcc;">{meaning}</span>
          </div>
          <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                      color:#3a5080;margin-bottom:0.4rem;">{formula}</div>
          <div style="font-size:0.83rem;color:#7a8fb0;">{explain}</div>
        </div>
        """, unsafe_allow_html=True)

        fig, axes = dark_fig(1, 3, figsize=(15, 4.5))
        titles = [
            f"{abbr} {year_before} (Baseline)",
            f"{abbr} {year_after} (Analysis)",
            f"Δ {abbr} (Change — mining signal)",
        ]
        cmaps = [cmap_b, cmap_a, cmap_d]
        arrays = [arr_b, arr_a, arr_d]
        for ax, arr, cmap_, title in zip(axes, arrays, cmaps, titles):
            valid = arr[~np.isnan(arr)]
            vmin_, vmax_ = (np.nanpercentile(arr, 2), np.nanpercentile(arr, 98)) if len(valid) else (0, 1)
            im = ax.imshow(arr, cmap=cmap_, vmin=vmin_, vmax=vmax_)
            ax.set_title(title, color="#e8f0fe", fontsize=10, pad=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="#7a8fb0")
            plt.setp(plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.get_ticklabels(), color="#7a8fb0")

        fig.suptitle(f"{abbr} — {full} · {year_before} vs {year_after}",
                     color="#e8f0fe", fontsize=12, fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

        st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — FUSION MAP
# ══════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown('<div class="section-header">Multi-Index <span>Fusion</span> Mining Probability</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:8px;
                padding:1rem 1.2rem;margin-bottom:1rem;font-family:'Space Mono',monospace;
                font-size:0.78rem;color:#7a8fb0;line-height:1.7;">
    <strong style="color:{ACCENT};">Fusion Formula</strong><br>
    Mining Score = {w_ndvi:.2f}·ΔNDVI + {w_bsi:.2f}·ΔBSI + {w_ndwi:.2f}·ΔNDWI + {w_nbr:.2f}·ΔNBR<br>
    All delta maps are normalised [0,1] before fusion. Threshold: <strong style="color:#e8f0fe;">{mining_thresh:.2f}</strong>
    </div>
    """, unsafe_allow_html=True)

    # ── Top row: 4 deltas ──────────────────────────────────────────────────
    fig, axes = dark_fig(1, 4, figsize=(18, 4.5))
    delta_info = [
        (d_ndvi, "RdYlGn_r", f"ΔNDVI\n(Vegetation Loss)\nWeight: {w_ndvi:.0%}"),
        (d_bsi,  "OrRd",     f"ΔBSI\n(Bare Soil Gain)\nWeight: {w_bsi:.0%}"),
        (d_ndwi, "RdBu",     f"ΔNDWI\n(Water Turbidity)\nWeight: {w_ndwi:.0%}"),
        (d_nbr,  "YlOrRd",   f"ΔNBR\n(Land Disturbance)\nWeight: {w_nbr:.0%}"),
    ]
    for ax, (arr, cmap_, title) in zip(axes, delta_info):
        im = ax.imshow(arr, cmap=cmap_, vmin=0, vmax=1)
        ax.set_title(title, color="#e8f0fe", fontsize=10, pad=6)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Component Delta Maps — Input to Fusion", color="#e8f0fe", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig_to_st(fig)

    # ── Bottom: Full fusion map ────────────────────────────────────────────
    fig2, ax2 = dark_fig(1, 1, figsize=(10, 9))
    score_display = np.ma.masked_where(mining_score < 0.2, mining_score)
    im2 = ax2.imshow(score_display, cmap="YlOrRd", vmin=0.2, vmax=1.0, alpha=0.9)
    ax2.imshow(ndvi_a, cmap="Greys_r", alpha=0.35, vmin=0, vmax=1)

    try:
        contours = ax2.contour(mining_score,
                               levels=[0.35, mining_thresh, 0.65],
                               colors=["#ffff00", "#ff8800", "#ff0000"],
                               linewidths=[0.8, 1.4, 2.0], alpha=0.95)
        ax2.clabel(contours, fmt={0.35:"35%", mining_thresh:f"{mining_thresh:.0%}", 0.65:"65%"},
                   colors="white", fontsize=9)
    except Exception:
        pass

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.035, pad=0.02)
    cbar2.set_label("Mining Probability Score", color="#e8f0fe", fontsize=11)
    plt.setp(cbar2.ax.yaxis.get_ticklabels(), color="#7a8fb0")

    stats_txt = (
        f"Affected Area (>{mining_thresh:.0%}) : {affected_ha:.0f} ha\n"
        f"Critical Zone (>65%)    : {critical_ha:.0f} ha\n"
        f"Peak Score              : {peak_score:.3f}\n"
        f"Resolution              : {resolution}m · Sentinel-2 L2A"
    )
    ax2.text(0.02, 0.02, stats_txt, transform=ax2.transAxes, color="white",
             fontsize=10, va="bottom", ha="left", family="monospace",
             bbox=dict(facecolor=BG, edgecolor=BORDER, alpha=0.92, boxstyle="round,pad=0.5"))
    ax2.set_title(f"Fused Mining Probability · {year_before} → {year_after}",
                  color="#e8f0fe", fontsize=13, fontweight="bold", pad=12)
    ax2.axis("off")
    fig2.tight_layout()
    fig_to_st(fig2)

    col1, col2, col3 = st.columns(3)
    col1.metric("Pixels above threshold", f"{int(mining_mask.sum()):,}")
    col2.metric("Affected Area",          f"{affected_ha:.1f} ha")
    col3.metric("Peak Mining Score",      f"{peak_score:.3f}")

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — LAND COVER CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">K-Means <span>Land Cover</span> Classification</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:8px;
                padding:1rem 1.2rem;margin-bottom:1rem;font-size:0.83rem;color:#7a8fb0;">
    K-Means clustering ({n_clusters} clusters) applied to all 4 spectral indices simultaneously.
    The same model is fit on <strong style="color:#e8f0fe;">{year_after}</strong> data and applied to
    <strong style="color:#e8f0fe;">{year_before}</strong> — enabling direct comparison.
    Clusters are auto-labelled by dominant index signature.
    </div>
    """, unsafe_allow_html=True)

    # Remap
    remap = {}
    sorted_keys = sorted(cluster_names.keys())
    for i, k in enumerate(sorted_keys):
        remap[k] = i
    remap_fn  = np.vectorize(remap.get)
    display_a = remap_fn(labels_a)
    display_b = remap_fn(labels_b)

    colors_list = [cluster_names[k][1] for k in sorted_keys]
    cmap_lc = ListedColormap(colors_list)

    fig, axes = dark_fig(1, 2, figsize=(16, 7))
    for ax, display, year in zip(axes, [display_b, display_a], [year_before, year_after]):
        ax.imshow(display, cmap=cmap_lc, vmin=0, vmax=len(sorted_keys)-1, interpolation="nearest")
        ax.set_title(f"Land Cover Classification — {year}", color="#e8f0fe", fontsize=13, fontweight="bold")
        ax.axis("off")

    patches = [mpatches.Patch(color=cluster_names[k][1], label=cluster_names[k][0])
               for k in sorted_keys]
    axes[1].legend(handles=patches, loc="lower left", facecolor=BG2,
                   labelcolor="#e8f0fe", fontsize=11, edgecolor=BORDER)

    mining_2019_ha = (labels_b == mining_cls).sum() * px_area_ha
    mining_2023_ha = (labels_a == mining_cls).sum() * px_area_ha
    forest_2019_ha = (labels_b == forest_cls).sum() * px_area_ha
    forest_2023_ha = (labels_a == forest_cls).sum() * px_area_ha

    summary = (
        f"⛏ Mining/Bare Soil\n"
        f"  {year_before}: {mining_2019_ha:.0f} ha\n"
        f"  {year_after}: {mining_2023_ha:.0f} ha\n"
        f"  Δ: +{mining_2023_ha - mining_2019_ha:.0f} ha\n\n"
        f"🌲 Forest/Vegetation\n"
        f"  {year_before}: {forest_2019_ha:.0f} ha\n"
        f"  {year_after}: {forest_2023_ha:.0f} ha\n"
        f"  Δ: {forest_2023_ha - forest_2019_ha:.0f} ha"
    )
    axes[1].text(0.98, 0.98, summary, transform=axes[1].transAxes,
                 color="white", fontsize=10, va="top", ha="right", family="monospace",
                 bbox=dict(facecolor=BG2, edgecolor=RED, alpha=0.95, boxstyle="round,pad=0.6"))

    fig.suptitle("Land Cover: What It WAS vs What It BECAME", color="#e8f0fe",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig_to_st(fig)

    # ── Change table ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:1.5rem;">Land Cover <span>Change Table</span></div>', unsafe_allow_html=True)
    for k in sorted_keys:
        name, color = cluster_names[k]
        area_b = (labels_b == k).sum() * px_area_ha
        area_a = (labels_a == k).sum() * px_area_ha
        change = area_a - area_b
        sign   = "+" if change >= 0 else ""
        trend  = "▲" if change > 50 else ("▼" if change < -50 else "→")
        c_color = RED if (change > 50 and "Mining" in name) else GREEN if change < -50 else "#7a8fb0"
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;padding:0.7rem 1rem;
                    background:{BG2};border:1px solid {BORDER};border-radius:6px;
                    margin-bottom:0.4rem;">
          <div style="width:14px;height:14px;border-radius:3px;background:{color};flex-shrink:0;"></div>
          <div style="font-family:'Syne',sans-serif;font-size:0.95rem;font-weight:600;
                      color:#e8f0fe;flex:1;">{name}</div>
          <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#7a8fb0;width:100px;">
            {year_before}: <strong>{area_b:.0f} ha</strong></div>
          <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#7a8fb0;width:100px;">
            {year_after}: <strong>{area_a:.0f} ha</strong></div>
          <div style="font-family:'Space Mono',monospace;font-size:0.85rem;
                      color:{c_color};font-weight:700;width:100px;">{trend} {sign}{change:.0f} ha</div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — TEMPORAL ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">Temporal <span>Anomaly</span> Detection</div>', unsafe_allow_html=True)

    if not temporal_data:
        st.warning("No temporal data available — check year selections and cloud cover.")
    else:
        t_colors = []
        for s in t_scores:
            if s > t_mean + 2 * t_std:
                t_colors.append(RED)
            elif s > t_thresh:
                t_colors.append(ORANGE)
            else:
                t_colors.append(GREEN)

        # ── Line + Bar chart ──────────────────────────────────────────────
        fig, axes = dark_fig(1, 2, figsize=(16, 6))

        # Left: line chart
        ax = axes[0]
        ax.fill_between(t_years, t_mean - t_std, t_mean + t_std,
                        alpha=0.15, color=GREEN, label="Normal Range (±1σ)")
        ax.axhline(t_thresh, color=RED, linestyle="--", linewidth=1.5,
                   label=f"Alert Threshold (μ+1σ = {t_thresh:.4f})")
        ax.axhline(t_mean, color=GREEN, linestyle=":", linewidth=1.2,
                   label=f"Baseline (μ = {t_mean:.4f})")
        ax.plot(t_years, t_scores, color="white", linewidth=2.5,
                marker="o", markersize=0, zorder=3)
        for yr, s, c in zip(t_years, t_scores, t_colors):
            ax.scatter(yr, s, color=c, s=180, zorder=5, edgecolors="white", linewidth=1.5)
            ax.annotate(f"{s:.4f}", (yr, s), textcoords="offset points",
                        xytext=(0, 14), color="white", fontsize=9, ha="center")
        ax.set_xticks(t_years)
        ax.set_xticklabels([str(y) for y in t_years], color="#7a8fb0", fontsize=11)
        ax.set_xlabel("Year", color="#7a8fb0", fontsize=12)
        ax.set_ylabel("Mean Mining Score", color="#7a8fb0", fontsize=12)
        ax.set_title("Yearly Mining Score Trend", color="#e8f0fe", fontsize=12, fontweight="bold")
        ax.legend(facecolor=BG2, labelcolor="#e8f0fe", fontsize=9)
        ax.grid(alpha=0.1, color="white")

        # Right: bar chart
        ax2 = axes[1]
        bars = ax2.bar(t_years, t_scores, color=t_colors, edgecolor="white", linewidth=0.8, width=0.6)
        ax2.axhline(t_thresh, color=RED, linestyle="--", linewidth=1.5,
                    label=f"Threshold = {t_thresh:.4f}")
        ax2.axhline(t_mean, color=GREEN, linestyle=":", linewidth=1.2,
                    label=f"Baseline = {t_mean:.4f}")
        for bar, s in zip(bars, t_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                     f"{s:.4f}", ha="center", va="bottom", color="white", fontsize=9)
        ax2.set_xticks(t_years)
        ax2.set_xticklabels([str(y) for y in t_years], color="#7a8fb0", fontsize=11)
        ax2.set_xlabel("Year", color="#7a8fb0", fontsize=12)
        ax2.set_ylabel("Mean Mining Score", color="#7a8fb0", fontsize=12)
        ax2.set_title("Year-on-Year Comparison", color="#e8f0fe", fontsize=12, fontweight="bold")
        ax2.legend(facecolor=BG2, labelcolor="#e8f0fe", fontsize=9)
        ax2.grid(alpha=0.1, color="white", axis="y")

        fig.suptitle("⚠️  Temporal Anomaly Detection — Illegal Mining",
                     color="#e8f0fe", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

        # ── Alert summary ─────────────────────────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Year-by-Year <span>Alert Report</span></div>', unsafe_allow_html=True)
        for yr, s, c in zip(t_years, t_scores, t_colors):
            if s > t_mean + 2*t_std:
                status = "🔴 CRITICAL — Likely Active Mining"
                cls    = "alert-critical"
            elif s > t_thresh:
                status = "🟡 SUSPECTED — Above Baseline"
                cls    = "alert-warning"
            else:
                status = "🟢 NORMAL — Within Expected Range"
                cls    = "alert-ok"
            st.markdown(f"""
            <div class="{cls}">
              <strong>{yr}</strong> &nbsp;|&nbsp; Score: {s:.4f} &nbsp;|&nbsp; {status}
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:8px;
                    padding:1rem 1.2rem;margin-top:1rem;font-family:'Space Mono',monospace;
                    font-size:0.78rem;color:#7a8fb0;line-height:1.8;">
        <strong style="color:{ACCENT};">Statistical Summary</strong><br>
        Baseline Mean   : {t_mean:.4f}<br>
        Std Deviation   : {t_std:.4f}<br>
        Alert Threshold : {t_thresh:.4f}<br>
        Score {t_years[0]}→{t_years[-1]} : {t_scores[0]:.4f} → {t_scores[-1]:.4f}
        ({'+' if t_scores[-1]>t_scores[0] else ''}{t_scores[-1]-t_scores[0]:.4f})<br>
        Anomalous Years : {sum(s > t_thresh for s in t_scores)} / {len(t_scores)}
        </div>
        """, unsafe_allow_html=True)

        # ── NDVI over time ─────────────────────────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:2rem;">NDVI Trend — <span>Vegetation Health Over Time</span></div>', unsafe_allow_html=True)
        t_ndvis = [temporal_data[y]["ndvi"] for y in t_years]
        fig3, ax3 = dark_fig(1, 1, figsize=(12, 4))
        ax3.fill_between(t_years, [n - 0.02 for n in t_ndvis], [n + 0.02 for n in t_ndvis],
                         alpha=0.1, color=GREEN)
        ax3.plot(t_years, t_ndvis, color=GREEN, linewidth=2.5, marker="o", markersize=10,
                 markerfacecolor=GREEN, markeredgecolor="white", markeredgewidth=1.5)
        for yr, n in zip(t_years, t_ndvis):
            ax3.annotate(f"{n:.3f}", (yr, n), textcoords="offset points",
                         xytext=(0, 12), color="white", fontsize=9, ha="center")
        ax3.set_xticks(t_years)
        ax3.set_xticklabels([str(y) for y in t_years], color="#7a8fb0")
        ax3.set_xlabel("Year", color="#7a8fb0")
        ax3.set_ylabel("Mean NDVI", color="#7a8fb0")
        ax3.set_title("Mean NDVI Over Time — Declining NDVI = Vegetation Destruction", color="#e8f0fe", fontweight="bold")
        ax3.grid(alpha=0.1, color="white")
        fig3.tight_layout()
        fig_to_st(fig3)

# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — NDVI ANIMATION
# ══════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">NDVI <span>Time-Lapse</span> Animation</div>', unsafe_allow_html=True)

    if not temporal_data:
        st.warning("No temporal data for animation.")
    else:
        st.markdown(f"""
        <div style="background:{BG2};border:1px solid {BORDER};border-radius:8px;
                    padding:1rem 1.2rem;margin-bottom:1rem;font-size:0.83rem;color:#7a8fb0;">
        Animated NDVI map cycling through {', '.join(str(y) for y in t_years)}.
        Watch the vegetation (green) disappear over the years — a direct visual indicator of mining expansion.
        </div>
        """, unsafe_allow_html=True)

        # Create a static multi-panel instead of animation (Streamlit-compatible)
        n_years = len(t_years)
        fig, axes = plt.subplots(1, n_years, figsize=(4 * n_years, 4.5))
        fig.patch.set_facecolor(BG)
        if n_years == 1:
            axes = [axes]

        for ax, yr in zip(axes, t_years):
            ndvi_map = temporal_data[yr].get("ndvi_map")
            if ndvi_map is None:
                ax.set_visible(False)
                continue
            ax.set_facecolor(BG2)
            im = ax.imshow(ndvi_map, cmap="RdYlGn", vmin=-0.3, vmax=0.8)
            mean_v = temporal_data[yr]["ndvi"]
            status = "🟢 Healthy" if mean_v > 0.3 else "🔴 Degraded"
            ax.set_title(f"{yr}\nNDVI: {mean_v:.3f}\n{status}",
                         color="#e8f0fe", fontsize=10, fontweight="bold")
            ax.axis("off")

        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, label="NDVI")
        fig.suptitle(f"NDVI Time-Lapse — {t_years[0]} to {t_years[-1]}",
                     color="#e8f0fe", fontsize=13, fontweight="bold")
        fig.tight_layout()
        fig_to_st(fig)

        # ── Year-by-year NDVI detailed maps ───────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Per-Year <span>NDVI Maps</span></div>', unsafe_allow_html=True)
        for yr in t_years:
            ndvi_map = temporal_data[yr].get("ndvi_map")
            if ndvi_map is None:
                continue
            with st.expander(f"📅 {yr} — Mean NDVI: {temporal_data[yr]['ndvi']:.3f}"):
                fig_y, ax_y = dark_fig(1, 1, figsize=(8, 6))
                im_y = ax_y.imshow(ndvi_map, cmap="RdYlGn", vmin=-0.3, vmax=0.8)
                ax_y.set_title(f"NDVI — {yr}", color="#e8f0fe", fontsize=13, fontweight="bold")
                ax_y.axis("off")
                plt.colorbar(im_y, ax=ax_y, fraction=0.046, pad=0.04)
                fig_y.tight_layout()
                fig_to_st(fig_y)

# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — FALSE POSITIVE FILTERING
# ══════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">False Positive <span>Filtering</span></div>', unsafe_allow_html=True)

    if osm_ok:
        st.markdown(f"""
        <div class="alert-ok">
        ✅ OpenStreetMap data fetched via Overpass API — {len(osm_data)} land-use features loaded.
        Residential, farmland, industrial, and urban areas are masked from the mining probability map.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="alert-warning">
        ⚠️ Could not reach Overpass API — false positive filtering was skipped.
        The filtered map will match the raw map.
        </div>
        """, unsafe_allow_html=True)

    fig, axes = dark_fig(1, 2, figsize=(16, 7))
    for ax, score, title in zip(
        axes,
        [mining_score, mining_score_filtered],
        [f"Before Filter — {affected_ha:.0f} ha flagged",
         f"After Filter — {filtered_ha:.0f} ha ({removed_ha:.0f} ha removed)"]
    ):
        ax.imshow(ndvi_a, cmap="Greys_r", alpha=0.35, vmin=0, vmax=1)
        sd = np.ma.masked_where(score < 0.2, score)
        im = ax.imshow(sd, cmap="YlOrRd", vmin=0.2, vmax=1.0, alpha=0.9)
        ax.set_title(title, color="#e8f0fe", fontsize=12, fontweight="bold")
        ax.axis("off")

    cbar = plt.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02)
    cbar.set_label("Mining Probability", color="#e8f0fe", fontsize=10)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#7a8fb0")

    fig.suptitle("False Positive Filtering — Urban / Agricultural Areas Excluded",
                 color="#e8f0fe", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig_to_st(fig)

    st.markdown(f"""
    <div style="background:{BG2};border:1px solid {BORDER};border-radius:8px;
                padding:1rem 1.2rem;margin-top:1rem;font-family:'Space Mono',monospace;
                font-size:0.8rem;color:#7a8fb0;line-height:1.8;">
    <strong style="color:{ACCENT};">Filter Summary</strong><br>
    Raw flagged area   : {affected_ha:.0f} ha<br>
    OSM features found : {len(osm_data) if osm_ok else 'N/A'}<br>
    Area removed (FP)  : {removed_ha:.0f} ha<br>
    Final estimate     : <strong style="color:#e8f0fe;">{filtered_ha:.0f} ha</strong><br>
    FP removal rate    : {(removed_ha/affected_ha*100) if affected_ha > 0 else 0:.1f}%
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — SCIENTIFIC PROBABILITY MAP
# ══════════════════════════════════════════════════════════════════════════
with tabs[6]:
    st.markdown('<div class="section-header">Scientific <span>Probability Map</span> with Contours</div>', unsafe_allow_html=True)

    fig, ax = dark_fig(1, 1, figsize=(10, 9))
    ax.imshow(ndvi_a, cmap="Greys_r", alpha=0.4, vmin=0, vmax=1)
    score_display = np.ma.masked_where(mining_score < 0.2, mining_score)
    im = ax.imshow(score_display, cmap="YlOrRd", vmin=0.2, vmax=1.0, alpha=0.9)

    try:
        contours = ax.contour(mining_score,
                              levels=[0.35, 0.50, 0.65, 0.80],
                              colors=["#ffff00", "#ff8800", "#ff0000", "#cc0000"],
                              linewidths=[0.8, 1.2, 1.8, 2.5], alpha=0.95)
        ax.clabel(contours, fmt={0.35:"35%",0.50:"50%",0.65:"65%",0.80:"80%"},
                  colors="white", fontsize=9)
    except Exception:
        pass

    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Mining Probability Score", color="#e8f0fe", fontsize=11)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#7a8fb0")

    px = px_area_ha
    a35 = (mining_score > 0.35).sum() * px
    a50 = (mining_score > 0.50).sum() * px
    a65 = (mining_score > 0.65).sum() * px

    stats_txt = (
        f"Affected  (>35%) : {a35:.0f} ha\n"
        f"Probable  (>50%) : {a50:.0f} ha\n"
        f"Critical  (>65%) : {a65:.0f} ha\n"
        f"Peak Score        : {peak_score:.3f}\n"
        f"Resolution        : {resolution}m · Sentinel-2 L2A\n"
        f"Weights           : NDVI={w_ndvi} BSI={w_bsi}\n"
        f"                    NDWI={w_ndwi} NBR={w_nbr}"
    )
    ax.text(0.02, 0.02, stats_txt, transform=ax.transAxes, color="white",
            fontsize=9, va="bottom", ha="left", family="monospace",
            bbox=dict(facecolor=BG, edgecolor=BORDER, alpha=0.92, boxstyle="round,pad=0.5"))

    ax.set_title(f"Mining Probability Map — Sentinel-2 Multi-Index Fusion\n{year_before} → {year_after}",
                 color="#e8f0fe", fontsize=13, fontweight="bold", pad=12)
    ax.axis("off")
    fig.tight_layout()
    fig_to_st(fig)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Suspected (>35%)", f"{a35:.0f} ha")
    col2.metric("Probable (>50%)",  f"{a50:.0f} ha")
    col3.metric("Critical (>65%)",  f"{a65:.0f} ha")
    col4.metric("Peak Score",       f"{peak_score:.3f}")

    # ── Histogram of mining scores ─────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:1.5rem;">Score <span>Distribution</span></div>', unsafe_allow_html=True)
    fig_h, ax_h = dark_fig(1, 1, figsize=(10, 4))
    flat = mining_score.flatten()
    flat = flat[flat > 0.05]
    ax_h.hist(flat, bins=80, color=ACCENT, edgecolor=BG, linewidth=0.3, alpha=0.85)
    ax_h.axvline(mining_thresh, color=ORANGE, linestyle="--", linewidth=2,
                 label=f"Threshold ({mining_thresh:.2f})")
    ax_h.axvline(0.65, color=RED, linestyle="--", linewidth=2, label="Critical (0.65)")
    ax_h.set_xlabel("Mining Score", color="#7a8fb0")
    ax_h.set_ylabel("Pixel Count", color="#7a8fb0")
    ax_h.set_title("Distribution of Mining Probability Scores Across All Pixels",
                   color="#e8f0fe", fontweight="bold")
    ax_h.legend(facecolor=BG2, labelcolor="#e8f0fe")
    ax_h.grid(alpha=0.1, color="white")
    fig_h.tight_layout()
    fig_to_st(fig_h)

# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — REPORT
# ══════════════════════════════════════════════════════════════════════════
with tabs[7]:
    st.markdown('<div class="section-header">📋 Analyst <span>Report</span></div>', unsafe_allow_html=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    anomalous_years = sum(s > t_thresh for s in t_scores) if temporal_data else "N/A"

    report_text = f"""
╔══════════════════════════════════════════════════════════════════════════╗
║             SENTINELWATCH — ILLEGAL MINING DETECTION REPORT             ║
║                    Generated: {timestamp}                    ║
╚══════════════════════════════════════════════════════════════════════════╝

AREA OF INTEREST
  Bounding Box : {lon_min:.4f},{lat_min:.4f} → {lon_max:.4f},{lat_max:.4f}
  Analysis     : {year_before} (baseline) → {year_after} (analysis)
  Imagery      : Sentinel-2 L2A via Microsoft Planetary Computer
  Resolution   : {resolution}m per pixel

RISK ASSESSMENT
  Risk Level   : {risk_level}
  Peak Score   : {peak_score:.4f}

AREA ESTIMATES
  Affected (>{mining_thresh:.0%})   : {affected_ha:.1f} hectares
  Probable (>50%)    : {(mining_score > 0.50).sum() * px_area_ha:.1f} hectares
  Critical (>65%)    : {critical_ha:.1f} hectares
  FP-Filtered        : {filtered_ha:.1f} hectares (after OSM masking)
  FP Removed         : {removed_ha:.1f} hectares

LAND COVER CHANGE
  Mining/Bare Soil   : {mining_2019_ha:.0f} ha ({year_before}) → {mining_2023_ha:.0f} ha ({year_after})  [Δ +{mining_2023_ha - mining_2019_ha:.0f} ha]
  Forest/Vegetation  : {forest_2019_ha:.0f} ha ({year_before}) → {forest_2023_ha:.0f} ha ({year_after})  [Δ {forest_2023_ha - forest_2019_ha:.0f} ha]

SPECTRAL INDICES (Analysis Year {year_after})
  NDVI (mean)  : {np.nanmean(ndvi_a):.4f}  [healthy vegetation > 0.4]
  BSI  (mean)  : {np.nanmean(bsi_a):.4f}   [bare soil > 0.0]
  NDWI (mean)  : {np.nanmean(ndwi_a):.4f}  [water presence > 0.0]
  NBR  (mean)  : {np.nanmean(nbr_a):.4f}   [undisturbed land > 0.3]

FUSION WEIGHTS
  NDVI : {w_ndvi:.2f}  |  BSI : {w_bsi:.2f}  |  NDWI : {w_ndwi:.2f}  |  NBR : {w_nbr:.2f}

TEMPORAL ANALYSIS ({', '.join(str(y) for y in t_years) if temporal_data else 'N/A'})
  Baseline Mean    : {t_mean:.4f if temporal_data else 'N/A'}
  Std Deviation    : {t_std:.4f if temporal_data else 'N/A'}
  Alert Threshold  : {t_thresh:.4f if temporal_data else 'N/A'}
  Anomalous Years  : {anomalous_years} / {len(t_years) if temporal_data else 'N/A'}

METHODOLOGY
  This analysis uses multi-spectral change detection combining four
  vegetation/soil/water indices from Sentinel-2 Level-2A imagery.
  K-Means clustering (k={n_clusters}) provides unsupervised land cover
  classification. OpenStreetMap data filters known urban/agricultural areas
  to reduce false positives. Temporal anomaly detection uses 1σ thresholding.

DISCLAIMER
  This tool provides satellite-based evidence of land surface change
  consistent with illegal mining activity. Results should be corroborated
  with field verification, legal analysis, and additional data sources
  before legal or regulatory action. All data is openly sourced.

SOURCE CODE & DATA
  Imagery  : ESA Copernicus Sentinel-2 (Free & Open)
  Catalog  : Microsoft Planetary Computer (https://planetarycomputer.microsoft.com)
  OSM Data : © OpenStreetMap contributors
  App      : SentinelWatch (Open Source, MIT License)

══════════════════════════════════════════════════════════════════════════
    """

    st.code(report_text, language="text")

    # Download button
    st.download_button(
        label="⬇️  Download Full Report (.txt)",
        data=report_text,
        file_name=f"sentinelwatch_report_{year_before}_{year_after}_{timestamp[:10]}.txt",
        mime="text/plain",
    )

    # ── Methodology explanation ────────────────────────────────────────────
    st.markdown('<div class="section-header" style="margin-top:1.5rem;">📖 Methodology <span>Guide</span></div>', unsafe_allow_html=True)
    methodology_items = [
        ("Why Sentinel-2?",
         "ESA's Sentinel-2 satellites provide free 10–60m imagery every 5 days globally. "
         "Their multispectral bands (Red, NIR, SWIR, Green) are specifically designed for land monitoring."),
        ("What does the score mean?",
         f"A score above {mining_thresh:.2f} indicates significant change consistent with mining. "
         "Above 0.65 is high confidence. The score combines NDVI loss (35%), BSI increase (30%), "
         "NDWI change (20%), and NBR change (15%)."),
        ("Limitations",
         "Cloud cover, seasonal vegetation changes, and agricultural clearing can cause false positives. "
         "Always use FP filtering and cross-reference with ground truth. This tool provides evidence, "
         "not legal proof."),
        ("Recommended Use",
         "Compare a year known to be undisturbed (baseline) with a suspect year. "
         "Use the temporal analysis to narrow down when activity began. "
         "Export results and share with environmental journalists, NGOs, or forest services."),
    ]
    for title, body in methodology_items:
        with st.expander(f"❓ {title}"):
            st.markdown(f'<div style="font-size:0.88rem;color:#7a8fb0;line-height:1.7;">{body}</div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown("""
<div style="text-align:center;font-family:'Space Mono',monospace;font-size:0.7rem;
            color:#3a5080;padding:1.5rem 0;line-height:2;">
🛰️ <strong style="color:#1e3058;">SENTINELWATCH</strong> — Open Source Illegal Mining Detection<br>
Data: ESA Copernicus Sentinel-2 · Microsoft Planetary Computer · OpenStreetMap<br>
Built for journalists, researchers, NGOs, and environmental defenders<br>
<span style="color:#1a2840;">All satellite data is freely available under Copernicus Open Access License</span>
</div>
""", unsafe_allow_html=True)
