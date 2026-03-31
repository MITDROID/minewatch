import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MineWatch — Illegal Mining Detection",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CUSTOM CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
    --bg:       #080b12;
    --surface:  #0f1420;
    --border:   #1e2535;
    --accent:   #ff4d1c;
    --accent2:  #ffb800;
    --green:    #00e676;
    --text:     #e8ecf4;
    --muted:    #6b7a99;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

h1, h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    color: var(--text) !important;
    letter-spacing: -0.5px;
}

.stButton > button {
    background: var(--accent) !important;
    color: white !important;
    border: none !important;
    border-radius: 4px !important;
    font-family: 'Space Mono', monospace !important;
    font-weight: 700 !important;
    font-size: 13px !important;
    letter-spacing: 1px !important;
    padding: 12px 24px !important;
    width: 100% !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #ff6a40 !important;
    transform: translateY(-1px) !important;
}

.metric-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-left: 3px solid var(--accent);
    border-radius: 6px;
    padding: 16px 20px;
    margin: 6px 0;
}

.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: var(--accent2);
    line-height: 1;
}

.metric-label {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 4px;
}

.critical-card {
    border-left-color: #ff1744 !important;
}

.safe-card {
    border-left-color: var(--green) !important;
}

.index-tag {
    display: inline-block;
    background: var(--border);
    color: var(--accent2);
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 3px;
    margin: 2px;
}

.dropped-tag {
    background: #1a0a0a;
    color: #ff4444;
}

.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 2px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 8px;
    margin: 20px 0 12px 0;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 42px;
    font-weight: 800;
    line-height: 1.1;
    background: linear-gradient(135deg, #ff4d1c, #ffb800);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--muted);
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-top: 4px;
}

.alert-box {
    background: #1a0a00;
    border: 1px solid var(--accent);
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 12px;
    color: var(--accent2);
    margin: 8px 0;
}

.info-box {
    background: #0a1020;
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 12px 16px;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
    margin: 8px 0;
}

/* Hide streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

[data-testid="stSlider"] > div { color: var(--text) !important; }

.stSelectbox > div > div {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stNumberInput > div > div > input {
    background: var(--surface) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}

/* Progress bar */
.stProgress > div > div > div {
    background: var(--accent) !important;
}

[data-testid="stExpander"] {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
}
</style>
""", unsafe_allow_html=True)


# ── HELPERS ────────────────────────────────────────────────────────────────
def get_band(stack, name):
    arr = stack.sel(band=name).values[0].astype(float)
    arr[arr == 0] = np.nan
    return arr

def norm(arr):
    arr = np.nan_to_num(arr, nan=0.0)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def compute_indices(stack):
    green = get_band(stack, "B03")
    red   = get_band(stack, "B04")
    re2   = get_band(stack, "B06")
    nir   = get_band(stack, "B08")
    nir_n = get_band(stack, "B8A")
    swir1 = get_band(stack, "B11")
    swir2 = get_band(stack, "B12")

    ndvi = (nir - red) / (nir + red + 1e-10)
    bsi  = ((swir1 + red) - (nir + green)) / ((swir1 + red) + (nir + green) + 1e-10)
    evi  = 2.5 * (nir - red) / (nir + 6*red - 7.5*green + 1 + 1e-10)
    fmi  = swir1 / (nir + 1e-10)
    cmi  = (swir1 - swir2) / (swir1 + swir2 + 1e-10)
    nbr  = (nir - swir1)   / (nir + swir1   + 1e-10)
    ndwi = (green - nir)   / (green + nir    + 1e-10)
    smci = (swir1 - nir_n) / (swir1 + nir_n + 1e-10)
    mbi  = (swir1 + red - nir) / (swir1 + red + nir + 1e-10)

    return dict(ndvi=ndvi, bsi=bsi, evi=evi, fmi=fmi,
                cmi=cmi, nbr=nbr, ndwi=ndwi, smci=smci, mbi=mbi)

BASE_WEIGHTS = {
    "ndvi": 0.18, "bsi": 0.15, "evi": 0.13, "fmi": 0.13,
    "cmi": 0.12,  "mbi": 0.10, "nbr": 0.08, "ndwi": 0.07, "smci": 0.04
}

CMAP_MAP = {
    "ndvi": 'RdYlGn_r', "bsi": 'OrRd',    "evi": 'RdYlGn_r',
    "fmi":  'Reds',      "cmi": 'copper',  "mbi": 'hot',
    "nbr":  'YlOrRd',    "ndwi": 'RdBu',   "smci": 'YlOrBr'
}

LABEL_MAP = {
    "ndvi": "NDVI — Vegetation Loss",    "bsi":  "BSI — Bare Soil Gain",
    "evi":  "EVI — Enhanced Veg Loss",   "fmi":  "FMI — Ferrous/Laterite",
    "cmi":  "CMI — Clay Mineral",        "mbi":  "MBI — Mining Brightness",
    "nbr":  "NBR — Land Disturbance",    "ndwi": "NDWI — Water Turbidity",
    "smci": "SMCI — Compacted Floor"
}

VARIANCE_THRESHOLD = 0.01


# ── SIDEBAR ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">⛏ Mine<br>Watch</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Sentinel-2 · Illegal Mining Detection</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div class="section-header">📍 Area of Interest</div>', unsafe_allow_html=True)

    preset = st.selectbox("Preset Region", [
        "Jharkhand Coal Belt (Default)",
        "Odisha Iron Ore Belt",
        "Custom Coordinates"
    ])

    if preset == "Jharkhand Coal Belt (Default)":
        lon_min, lon_max = 85.8, 86.2
        lat_min, lat_max = 23.5, 23.8
    elif preset == "Odisha Iron Ore Belt":
        lon_min, lon_max = 85.0, 85.5
        lat_min, lat_max = 22.0, 22.5
    else:
        col1, col2 = st.columns(2)
        with col1:
            lon_min = st.number_input("Lon Min", value=85.8, format="%.2f")
            lat_min = st.number_input("Lat Min", value=23.5, format="%.2f")
        with col2:
            lon_max = st.number_input("Lon Max", value=86.2, format="%.2f")
            lat_max = st.number_input("Lat Max", value=23.8, format="%.2f")

    st.markdown('<div class="section-header">📅 Time Period</div>', unsafe_allow_html=True)
    year_before = st.selectbox("Baseline Year", [2018, 2019, 2020], index=1)
    year_after  = st.selectbox("Analysis Year", [2021, 2022, 2023], index=2)

    st.markdown('<div class="section-header">⚙️ Parameters</div>', unsafe_allow_html=True)
    cloud_cover   = st.slider("Max Cloud Cover %", 5, 40, 20)
    threshold_35  = st.slider("Alert Threshold %", 20, 60, 35)
    resolution    = st.selectbox("Resolution (m)", [60, 30], index=0)

    st.markdown("---")
    run_btn = st.button("▶  RUN ANALYSIS")

    st.markdown('<div class="info-box">💡 First run takes ~3 min<br>Downloads Sentinel-2 tiles<br>from Planetary Computer</div>',
                unsafe_allow_html=True)


# ── MAIN PANEL ─────────────────────────────────────────────────────────────
st.markdown('<h1>Illegal Mining Detection</h1>', unsafe_allow_html=True)
st.markdown(f'<div class="hero-sub">Jharkhand, India &nbsp;·&nbsp; {year_before} → {year_after} &nbsp;·&nbsp; 9-Index Sentinel-2 Fusion</div>',
            unsafe_allow_html=True)
st.markdown("")

if not run_btn:
    # ── LANDING STATE ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">9</div>
            <div class="metric-label">Spectral Indices</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">60m</div>
            <div class="metric-label">Spatial Resolution</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">L2A</div>
            <div class="metric-label">Sentinel-2 Product</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How it works")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Indices used:**
        """)
        for k, v in LABEL_MAP.items():
            st.markdown(f'<span class="index-tag">{k.upper()}</span> {v}', unsafe_allow_html=True)

    with col_b:
        st.markdown("""
        **Pipeline:**
        1. Fetch fresh Sentinel-2 tiles from Planetary Computer
        2. Compute 9 spectral indices for baseline & analysis year
        3. Calculate change deltas between years
        4. Auto-diagnose flat/noisy indices and drop them
        5. Fuse surviving indices with dynamic weights
        6. Generate probability map + area statistics
        """)

    st.markdown('<div class="alert-box">⚠️  Configure your AOI and time period in the sidebar, then hit RUN ANALYSIS</div>',
                unsafe_allow_html=True)

else:
    # ── RUN PIPELINE ───────────────────────────────────────────────────────
    try:
        import pystac_client
        import planetary_computer
        import stackstac

        aoi = {
            "type": "Polygon",
            "coordinates": [[
                [lon_min, lat_min], [lon_max, lat_min],
                [lon_max, lat_max], [lon_min, lat_max],
                [lon_min, lat_min]
            ]]
        }

        progress = st.progress(0)
        status   = st.empty()

        # ── 1. Connect ─────────────────────────────────────────────────────
        status.markdown('<div class="alert-box">🛰  Connecting to Planetary Computer...</div>',
                        unsafe_allow_html=True)

        catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1",
            modifier=planetary_computer.sign_inplace,
        )
        progress.progress(10)

        # ── 2. Search ──────────────────────────────────────────────────────
        status.markdown('<div class="alert-box">🔍  Searching for Sentinel-2 scenes...</div>',
                        unsafe_allow_html=True)

        items_b = list(catalog.search(
            collections=["sentinel-2-l2a"], intersects=aoi,
            datetime=f"{year_before}-01-01/{year_before}-12-31",
            query={"eo:cloud_cover": {"lt": cloud_cover}},
        ).items())

        items_a = list(catalog.search(
            collections=["sentinel-2-l2a"], intersects=aoi,
            datetime=f"{year_after}-01-01/{year_after}-12-31",
            query={"eo:cloud_cover": {"lt": cloud_cover}},
        ).items())

        if not items_b or not items_a:
            st.error(f"No scenes found. Try increasing cloud cover threshold.")
            st.stop()

        progress.progress(25)
        status.markdown(f'<div class="alert-box">📦  Found {len(items_b)} before / {len(items_a)} after scenes. Loading bands...</div>',
                        unsafe_allow_html=True)

        # ── 3. Stack ───────────────────────────────────────────────────────
        ASSETS = ["B03", "B04", "B06", "B08", "B8A", "B11", "B12"]

        stack_b = stackstac.stack(
            items_b[:1], assets=ASSETS,
            resolution=resolution, epsg=32645, chunksize=512
        ).compute()

        progress.progress(50)

        stack_a = stackstac.stack(
            items_a[:1], assets=ASSETS,
            resolution=resolution, epsg=32645, chunksize=512
        ).compute()

        progress.progress(65)
        status.markdown('<div class="alert-box">🧮  Computing 9 spectral indices...</div>',
                        unsafe_allow_html=True)

        # ── 4. Indices ─────────────────────────────────────────────────────
        idx_b = compute_indices(stack_b)
        idx_a = compute_indices(stack_a)

        # Deltas
        delta_dir = {
            "ndvi": idx_b["ndvi"] - idx_a["ndvi"],
            "bsi":  idx_a["bsi"]  - idx_b["bsi"],
            "evi":  idx_b["evi"]  - idx_a["evi"],
            "fmi":  idx_a["fmi"]  - idx_b["fmi"],
            "cmi":  idx_a["cmi"]  - idx_b["cmi"],
            "nbr":  idx_b["nbr"]  - idx_a["nbr"],
            "ndwi": idx_b["ndwi"] - idx_a["ndwi"],
            "smci": idx_a["smci"] - idx_b["smci"],
            "mbi":  idx_a["mbi"]  - idx_b["mbi"],
        }

        progress.progress(75)
        status.markdown('<div class="alert-box">📊  Running signal diagnostics...</div>',
                        unsafe_allow_html=True)

        # ── 5. Diagnostic ──────────────────────────────────────────────────
        keep, drop_list = {}, []
        diag_rows = []

        for name, delta in delta_dir.items():
            d        = np.nan_to_num(delta, nan=0.0)
            variance = float(np.var(d))
            pos_frac = float((d > 0).mean())
            rng      = float(d.max() - d.min())
            kept     = variance >= VARIANCE_THRESHOLD

            diag_rows.append({
                "Index": name.upper(),
                "Variance": f"{variance:.5f}",
                "Pos Pixels": f"{pos_frac:.2f}",
                "Range": f"{rng:.4f}",
                "Status": "✓ KEEP" if kept else "✗ DROP"
            })

            if kept:
                keep[name] = norm(d)
            else:
                drop_list.append(name.upper())

        # Dynamic weights
        kept_w     = {k: BASE_WEIGHTS[k] for k in keep}
        total_w    = sum(kept_w.values())
        final_w    = {k: v / total_w for k, v in kept_w.items()}

        # Fused score
        h, w_px = list(keep.values())[0].shape
        score   = np.zeros((h, w_px))
        for name, d in keep.items():
            score += final_w[name] * d
        score = norm(score)

        progress.progress(88)

        # Stats
        affected = (score > threshold_35/100).sum() * (resolution * resolution / 10000)
        critical = (score > 0.65).sum()             * (resolution * resolution / 10000)
        peak     = float(score.max())

        progress.progress(100)
        status.empty()

        # ── RESULTS ────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📊 Results")

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{affected:.0f}</div>
                <div class="metric-label">Affected Area (ha)</div>
            </div>""", unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card critical-card">
                <div class="metric-value">{critical:.0f}</div>
                <div class="metric-label">Critical Zone (ha)</div>
            </div>""", unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{peak:.3f}</div>
                <div class="metric-label">Peak Score</div>
            </div>""", unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="metric-card safe-card">
                <div class="metric-value">{len(keep)}/{len(BASE_WEIGHTS)}</div>
                <div class="metric-label">Active Indices</div>
            </div>""", unsafe_allow_html=True)

        # Dropped indices
        if drop_list:
            tags = "".join([f'<span class="index-tag dropped-tag">{t}</span>' for t in drop_list])
            st.markdown(f'<div style="margin:8px 0">Dropped (flat signal): {tags}</div>',
                        unsafe_allow_html=True)

        st.markdown("---")

        # ── PROBABILITY MAP ────────────────────────────────────────────────
        st.markdown("### 🗺️ Mining Probability Map")

        ndvi_a = idx_a["ndvi"]
        fig, ax = plt.subplots(figsize=(10, 8))
        fig.patch.set_facecolor('#0f0f1a')
        ax.set_facecolor('#0f0f1a')

        ax.imshow(ndvi_a, cmap='Greys_r', alpha=0.4, vmin=0, vmax=1)
        display = np.ma.masked_where(score < 0.2, score)
        im = ax.imshow(display, cmap='YlOrRd', vmin=0.2, vmax=1.0, alpha=0.9)

        contours = ax.contour(score,
                              levels=[0.35, 0.50, 0.65],
                              colors=['#ffff00', '#ff8800', '#ff0000'],
                              linewidths=[0.8, 1.2, 1.6], alpha=0.9)
        ax.clabel(contours, fmt={0.35: '35%', 0.50: '50%', 0.65: '65%'},
                  colors='white', fontsize=9)

        cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label('Mining Probability', color='white', fontsize=11)
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

        kept_str = ' · '.join([k.upper() for k in keep])
        ax.set_title(f'Mining Probability Map\n{year_before} → {year_after}  |  {kept_str}',
                     color='white', fontsize=12, fontweight='bold', pad=12)
        ax.axis('off')

        stats_txt = (f"Affected (>{threshold_35}%) : {affected:.0f} ha\n"
                     f"Critical (>65%)  : {critical:.0f} ha\n"
                     f"Peak Score       : {peak:.3f}\n"
                     f"Resolution       : {resolution}m  |  S2-L2A")
        ax.text(0.02, 0.02, stats_txt, transform=ax.transAxes,
                color='white', fontsize=9, va='bottom', family='monospace',
                bbox=dict(facecolor='#0f0f1a', edgecolor='#444466',
                          alpha=0.9, boxstyle='round,pad=0.5'))

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── INDEX MAPS ─────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔬 Active Index Maps")

        n_keep = len(keep)
        ncols  = 3
        nrows  = (n_keep + ncols - 1) // ncols

        fig2, axes2 = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        fig2.patch.set_facecolor('#0f0f1a')

        axes_flat = axes2.flat if nrows > 1 else (axes2 if ncols > 1 else [axes2])
        for ax in axes_flat:
            ax.set_facecolor('#0f0f1a')
            ax.axis('off')

        for ax, (name, data) in zip(axes_flat, keep.items()):
            im2 = ax.imshow(data, cmap=CMAP_MAP[name])
            ax.set_title(LABEL_MAP[name], color='white', fontsize=11, fontweight='bold')
            ax.axis('off')
            cb = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color='white')
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

        # ── BEFORE / AFTER NDVI ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🌿 Vegetation Change (NDVI)")

        fig3, axes3 = plt.subplots(1, 2, figsize=(14, 6))
        fig3.patch.set_facecolor('#0f0f1a')

        for ax, ndvi, year_lbl in zip(axes3,
                                       [idx_b["ndvi"], idx_a["ndvi"]],
                                       [str(year_before), str(year_after)]):
            ax.set_facecolor('#0f0f1a')
            im3 = ax.imshow(ndvi, cmap='RdYlGn', vmin=-0.3, vmax=0.8)
            ax.set_title(f'NDVI {year_lbl}', color='white',
                         fontsize=13, fontweight='bold')
            ax.axis('off')
            cb = plt.colorbar(im3, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.yaxis.set_tick_params(color='white')
            plt.setp(cb.ax.yaxis.get_ticklabels(), color='white')

        plt.tight_layout()
        st.pyplot(fig3)
        plt.close()

        # ── DIAGNOSTIC TABLE ───────────────────────────────────────────────
        with st.expander("📋 Signal Diagnostic Report"):
            import pandas as pd
            df = pd.DataFrame(diag_rows)
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("**Final weights (after dropping flat indices):**")
            for k, v in final_w.items():
                bar = "█" * int(v * 100)
                st.markdown(f'`{k.upper():5s}  {v:.3f}  {bar}`')

    except ImportError as e:
        st.error(f"Missing dependency: {e}")
        st.code("pip install pystac-client planetary-computer stackstac", language="bash")

    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)
