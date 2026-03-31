import streamlit as st
import planetary_computer
import pystac_client
import stackstac
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import requests
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Mining Detection", layout="wide")

st.markdown("""
<style>
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    h1 { font-size: 1.6rem; font-weight: 600; }
    h2 { font-size: 1.2rem; font-weight: 500; border-bottom: 1px solid #e0e0e0; padding-bottom: 6px; }
    .stMetric label { font-size: 0.75rem; color: #666; }
    .stMetric value { font-size: 1.4rem; }
</style>
""", unsafe_allow_html=True)

st.title("Illegal Mining Detection — Jharkhand, India")
st.caption("Sentinel-2 Multi-Index Fusion | 2019 vs 2023")

AOI = {
    "type": "Polygon",
    "coordinates": [[
        [85.8, 23.5], [86.2, 23.5],
        [86.2, 23.8], [85.8, 23.8],
        [85.8, 23.5]
    ]]
}

YEARS = [2019, 2020, 2021, 2022, 2023]
PX_HA = 0.36

# ── Data loading ──────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

@st.cache_data(show_spinner=False)
def fetch_stack(year):
    catalog = get_catalog()
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        intersects=AOI,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": 20}},
    )
    items = list(search.items())
    if not items:
        return None
    stack = stackstac.stack(
        items[:1],
        assets=["B04", "B08", "B11", "B03"],
        resolution=60,
        epsg=32645,
        chunksize=512,
    ).compute()
    return stack

def compute_indices(stack):
    red   = stack.sel(band="B04").values[0].astype(float)
    nir   = stack.sel(band="B08").values[0].astype(float)
    swir  = stack.sel(band="B11").values[0].astype(float)
    green = stack.sel(band="B03").values[0].astype(float)
    for arr in [red, nir, swir, green]:
        arr[arr == 0] = np.nan
    ndvi = (nir - red)   / (nir + red   + 1e-10)
    bsi  = ((swir + red) - (nir + green)) / ((swir + red) + (nir + green) + 1e-10)
    ndwi = (green - nir) / (green + nir  + 1e-10)
    nbr  = (nir - swir)  / (nir + swir   + 1e-10)
    return ndvi, bsi, ndwi, nbr

def normalise(arr):
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, 0, None)
    return arr / arr.max() if arr.max() > 0 else arr

def compute_score(ndvi_b, bsi_b, ndwi_b, nbr_b, ndvi_a, bsi_a, ndwi_a, nbr_a):
    d_ndvi = normalise(ndvi_b - ndvi_a)
    d_bsi  = normalise(bsi_a  - bsi_b)
    d_ndwi = normalise(ndwi_b - ndwi_a)
    d_nbr  = normalise(nbr_b  - nbr_a)
    score  = 0.35*d_ndvi + 0.30*d_bsi + 0.20*d_ndwi + 0.15*d_nbr
    return score, d_ndvi, d_bsi, d_ndwi, d_nbr

# ── Load base data ─────────────────────────────────────────────────────────

with st.spinner("Loading satellite data..."):
    stack_2019 = fetch_stack(2019)
    stack_2023 = fetch_stack(2023)

if stack_2019 is None or stack_2023 is None:
    st.error("Could not load satellite data.")
    st.stop()

ndvi_b, bsi_b, ndwi_b, nbr_b = compute_indices(stack_2019)
ndvi_a, bsi_a, ndwi_a, nbr_a = compute_indices(stack_2023)
mining_score, d_ndvi, d_bsi, d_ndwi, d_nbr = compute_score(
    ndvi_b, bsi_b, ndwi_b, nbr_b,
    ndvi_a, bsi_a, ndwi_a, nbr_a
)
mining_mask = mining_score > 0.5

# ── Summary metrics ────────────────────────────────────────────────────────

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Affected Area", f"{mining_mask.sum() * PX_HA:.0f} ha")
c2.metric("Peak Score", f"{mining_score.max():.3f}")
c3.metric("Period", "2019 – 2023")
c4.metric("Resolution", "60 m")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────

tabs = st.tabs([
    "Multi-Index Fusion",
    "Probability Map",
    "Temporal Analysis",
    "Land Cover",
    "False Positive Filter",
])

# ── Tab 1: Multi-Index Fusion ──────────────────────────────────────────────

with tabs[0]:
    st.markdown("## Multi-index fusion")
    st.caption("Four spectral indices combined into a single mining probability score.")

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    panels = [
        (d_ndvi, "RdYlGn_r", "NDVI change\nvegetation loss"),
        (d_bsi,  "OrRd",     "BSI change\nbare soil gain"),
        (d_ndwi, "RdBu",     "NDWI change\nwater turbidity"),
        (d_nbr,  "YlOrRd",   "NBR change\nland disturbance"),
        (mining_score, "hot","Fused score\n(weighted sum)"),
    ]
    for ax, (data, cmap, title) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Fusion weights**")
    w1, w2, w3, w4 = st.columns(4)
    w1.metric("NDVI", "35%")
    w2.metric("BSI",  "30%")
    w3.metric("NDWI", "20%")
    w4.metric("NBR",  "15%")

# ── Tab 2: Probability Map ─────────────────────────────────────────────────

with tabs[1]:
    st.markdown("## Mining probability map")
    st.caption("Contour lines at 35%, 50%, and 65% confidence thresholds.")

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(ndvi_a, cmap="Greys_r", alpha=0.4, vmin=0, vmax=1)
    score_display = np.ma.masked_where(mining_score < 0.2, mining_score)
    im = ax.imshow(score_display, cmap="YlOrRd", vmin=0.2, vmax=1.0, alpha=0.9)
    contours = ax.contour(mining_score, levels=[0.35, 0.50, 0.65],
                          colors=["#ffff00", "#ff8800", "#ff0000"],
                          linewidths=[0.8, 1.2, 1.6])
    ax.clabel(contours, fmt={0.35: "35%", 0.50: "50%", 0.65: "65%"},
              colors="white", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Mining probability")
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    affected = (mining_score > 0.35).sum() * PX_HA
    critical  = (mining_score > 0.65).sum() * PX_HA
    ca, cb = st.columns(2)
    ca.metric("Area above 35% confidence", f"{affected:.0f} ha")
    cb.metric("Critical zone above 65%",   f"{critical:.0f} ha")

# ── Tab 3: Temporal Analysis ───────────────────────────────────────────────

with tabs[2]:
    st.markdown("## Temporal anomaly detection")
    st.caption("Year-on-year mining score and NDVI trend across 2019–2023.")

    with st.spinner("Loading yearly data..."):
        yearly_scores = []
        yearly_ndvi   = []
        loaded_years  = []

        for year in YEARS:
            s = fetch_stack(year)
            if s is None:
                continue
            n, b, w_, r = compute_indices(s)
            score_map = (
                0.35*normalise(n) + 0.30*normalise(b) +
                0.20*normalise(w_) + 0.15*normalise(r)
            )
            yearly_scores.append(float(np.nanmean(score_map)))
            yearly_ndvi.append(float(np.nanmean(np.nan_to_num(n))))
            loaded_years.append(year)

    scores = np.array(yearly_scores)
    mean_s = scores.mean()
    std_s  = scores.std()
    thresh = mean_s + std_s

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = ["#d62728" if s > mean_s + 2*std_s
              else "#ff7f0e" if s > thresh
              else "#2ca02c"
              for s in scores]

    axes[0].plot(loaded_years, scores, color="#333", linewidth=2,
                 marker="o", markersize=8, zorder=3)
    for yr, s, c in zip(loaded_years, scores, colors):
        axes[0].scatter(yr, s, color=c, s=100, zorder=4)
        axes[0].annotate(f"{s:.4f}", (yr, s),
                         textcoords="offset points", xytext=(0, 10),
                         fontsize=8, ha="center", color="#333")
    axes[0].fill_between(loaded_years, mean_s - std_s, mean_s + std_s,
                         alpha=0.15, color="green", label="Normal range")
    axes[0].axhline(thresh, color="red", linestyle="--",
                    linewidth=1.2, label=f"Alert threshold ({thresh:.4f})")
    axes[0].set_xticks(loaded_years)
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Mean mining score")
    axes[0].set_title("Mining score trend")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.2)

    axes[1].bar(loaded_years, yearly_ndvi, color="#4c72b0", width=0.6)
    axes[1].set_xticks(loaded_years)
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Mean NDVI")
    axes[1].set_title("Vegetation health trend")
    axes[1].grid(alpha=0.2, axis="y")
    for yr, v in zip(loaded_years, yearly_ndvi):
        axes[1].text(yr, v + 0.002, f"{v:.3f}",
                     ha="center", fontsize=8, color="#333")

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Year-by-year summary**")
    for yr, s in zip(loaded_years, scores):
        status = "Critical" if s > mean_s + 2*std_s \
            else "Suspected" if s > thresh \
            else "Normal"
        st.write(f"{yr} — Score: `{s:.4f}` — {status}")

# ── Tab 4: Land Cover Classification ──────────────────────────────────────

with tabs[3]:
    st.markdown("## Land cover classification")
    st.caption("K-Means clustering on spectral indices. Four land cover classes.")

    with st.spinner("Running classification..."):
        h, w = ndvi_a.shape
        feat_a = np.stack([np.nan_to_num(ndvi_a), np.nan_to_num(bsi_a),
                           np.nan_to_num(ndwi_a), np.nan_to_num(nbr_a)],
                          axis=-1).reshape(-1, 4)
        feat_b = np.stack([np.nan_to_num(ndvi_b), np.nan_to_num(bsi_b),
                           np.nan_to_num(ndwi_b), np.nan_to_num(nbr_b)],
                          axis=-1).reshape(-1, 4)

        km = KMeans(n_clusters=4, random_state=42, n_init=10)
        km.fit(feat_a)
        labels_a = km.predict(feat_a).reshape(h, w)
        labels_b = km.predict(feat_b).reshape(h, w)

        centers      = km.cluster_centers_
        forest_cls   = int(np.argmax(centers[:, 0]))
        mining_cls   = int(np.argmax(centers[:, 1]))
        water_cls    = int(np.argmax(centers[:, 2]))
        degraded_cls = list({0,1,2,3} - {forest_cls, mining_cls, water_cls})[0]

        remap    = {forest_cls: 0, water_cls: 1, degraded_cls: 2, mining_cls: 3}
        remap_fn = np.vectorize(remap.get)
        disp_a   = remap_fn(labels_a)
        disp_b   = remap_fn(labels_b)

    cmap_lc = ListedColormap(["#2d8a2d", "#3399ff", "#ff8800", "#ff2222"])
    class_names = {
        forest_cls:   ("Forest / Vegetation",       "#2d8a2d"),
        water_cls:    ("Water bodies",               "#3399ff"),
        degraded_cls: ("Degraded land",              "#ff8800"),
        mining_cls:   ("Active mining / Bare soil",  "#ff2222"),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, disp, year in zip(axes, [disp_b, disp_a], ["2019", "2023"]):
        ax.imshow(disp, cmap=cmap_lc, vmin=0, vmax=3, interpolation="nearest")
        ax.set_title(f"Land cover {year}", fontsize=12)
        ax.axis("off")
    patches = [mpatches.Patch(color=c, label=n)
               for _, (n, c) in class_names.items()]
    axes[1].legend(handles=patches, loc="lower left", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("**Area change summary**")
    for cls, (name, _) in class_names.items():
        a2019 = (labels_b == cls).sum() * PX_HA
        a2023 = (labels_a == cls).sum() * PX_HA
        delta = a2023 - a2019
        sign  = "+" if delta >= 0 else ""
        st.write(f"{name}: 2019 `{a2019:.0f} ha` → 2023 `{a2023:.0f} ha` ({sign}{delta:.0f} ha)")

# ── Tab 5: False Positive Filter ──────────────────────────────────────────

with tabs[4]:
    st.markdown("## False positive filtering")
    st.caption("Urban and agricultural areas removed using OpenStreetMap landuse data.")

    with st.spinner("Fetching landuse data..."):
        try:
            query = """
            [out:json][timeout:25];
            (
              way["landuse"~"residential|farmland|farm|industrial"](23.5,85.8,23.8,86.2);
              way["place"~"village|town|city|hamlet"](23.5,85.8,23.8,86.2);
            );
            out geom;
            """
            resp = requests.get(
                "http://overpass-api.de/api/interpreter",
                params={"data": query}, timeout=30
            )
            elements = resp.json().get("elements", [])
            h_, w_ = mining_score.shape
            fp_mask = np.zeros((h_, w_), dtype=bool)
            for el in elements:
                if "geometry" not in el:
                    continue
                for n in el["geometry"]:
                    col = int((n["lon"] - 85.8) / (86.2 - 85.8) * w_)
                    row = int((23.8 - n["lat"]) / (23.8 - 23.5) * h_)
                    if 0 <= row < h_ and 0 <= col < w_:
                        fp_mask[row, col] = True
            score_filtered = mining_score.copy()
            score_filtered[fp_mask] = 0
            osm_ok = True
        except Exception:
            osm_ok = False
            score_filtered = mining_score.copy()

    if not osm_ok:
        st.warning("OSM data unavailable. Showing unfiltered map.")

    before_ha  = (mining_score    > 0.35).sum() * PX_HA
    after_ha   = (score_filtered  > 0.35).sum() * PX_HA
    removed_ha = before_ha - after_ha

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, score, title in zip(
        axes,
        [mining_score, score_filtered],
        [f"Before filter\n{before_ha:.0f} ha flagged",
         f"After filter\n{after_ha:.0f} ha  ({removed_ha:.0f} ha removed)"]
    ):
        ax.imshow(ndvi_a, cmap="Greys_r", alpha=0.4, vmin=0, vmax=1)
        sd = np.ma.masked_where(score < 0.2, score)
        im = ax.imshow(sd, cmap="YlOrRd", vmin=0.2, vmax=1.0, alpha=0.9)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    plt.colorbar(im, ax=axes[1], fraction=0.035, pad=0.02,
                 label="Mining probability")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    fa, fb = st.columns(2)
    fa.metric("Before filter", f"{before_ha:.0f} ha")
    fb.metric("After filter",  f"{after_ha:.0f} ha",
              delta=f"-{removed_ha:.0f} ha false positives removed")
