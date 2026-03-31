import streamlit as st
import planetary_computer
import pystac_client
import stackstac
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import requests
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="MineWatch", layout="wide")

st.markdown("""
<style>
body, .stApp { background-color: #0e1117; color: #c9d1d9; }
[data-testid="stAppViewContainer"] { background-color: #0e1117; }
[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
h1,h2,h3 { color: #e6edf3 !important; }
.stMetric { background:#161b22; border:1px solid #30363d; border-radius:8px; padding:10px; }
.stMetric label { color:#8b949e !important; font-size:0.72rem; }
[data-testid="stMetricValue"] { color:#e6edf3 !important; }
.stTabs [data-baseweb="tab"] { color:#8b949e; background:#0e1117; }
.stTabs [aria-selected="true"] { color:#58a6ff !important; border-bottom:2px solid #58a6ff; }
hr { border-color:#30363d; }
p, .stMarkdown p { color:#8b949e; }
.stSelectbox>div>div, .stSlider { background:#161b22; }
</style>
""", unsafe_allow_html=True)

BG   = "#0e1117"
CARD = "#161b22"
PX_HA = 0.36
YEARS = [2019, 2020, 2021, 2022, 2023]
AOI = {
    "type": "Polygon",
    "coordinates": [[[85.8,23.5],[86.2,23.5],[86.2,23.8],[85.8,23.8],[85.8,23.5]]]
}

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": CARD,
    "axes.edgecolor": "#30363d", "axes.labelcolor": "#8b949e",
    "xtick.color": "#8b949e", "ytick.color": "#8b949e",
    "text.color": "#c9d1d9", "grid.color": "#30363d",
    "legend.facecolor": CARD, "legend.edgecolor": "#30363d",
})

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Configuration")
    st.markdown("---")
    region = st.selectbox("Region", [
        "Jharkhand Mining Belt, India",
        "Amazonas, Brazil",
        "Katanga, DR Congo",
        "Madre de Dios, Peru",
    ], index=0)
    year_range = st.select_slider("Analysis period",
        options=[2017,2018,2019,2020,2021,2022,2023], value=(2019,2023))
    st.selectbox("Satellite source", ["Sentinel-2 L2A (60m)","Landsat-9 (30m)"], index=0)
    st.selectbox("Detection model", ["Multi-Index Fusion","Single-Index NDVI"], index=0)
    st.markdown("---")
    run = st.button("Run analysis", use_container_width=True)
    st.caption("Demo uses Jharkhand 2019–2023.")

# ── Header ─────────────────────────────────────────────────────────────────
st.title("MineWatch — Illegal Mining Detection")
st.caption(f"Sentinel-2 Multi-Index Fusion  |  {region}  |  {year_range[0]} – {year_range[1]}")

# ── Data functions ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_catalog():
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )

@st.cache_data(show_spinner=False)
def fetch_stack(year):
    cat = get_catalog()
    items = list(cat.search(
        collections=["sentinel-2-l2a"], intersects=AOI,
        datetime=f"{year}-01-01/{year}-12-31",
        query={"eo:cloud_cover": {"lt": 20}},
    ).items())
    if not items:
        return None
    return stackstac.stack(items[:1], assets=["B04","B08","B11","B03"],
                           resolution=60, epsg=32645, chunksize=512).compute()

@st.cache_data(show_spinner=False)
def get_all_data():
    stacks = {}
    for y in YEARS:
        stacks[y] = fetch_stack(y)
    return stacks

def compute_indices(stack):
    r = stack.sel(band="B04").values[0].astype(float)
    n = stack.sel(band="B08").values[0].astype(float)
    s = stack.sel(band="B11").values[0].astype(float)
    g = stack.sel(band="B03").values[0].astype(float)
    for a in [r,n,s,g]: a[a==0] = np.nan
    ndvi = (n-r)/(n+r+1e-10)
    bsi  = ((s+r)-(n+g))/((s+r)+(n+g)+1e-10)
    ndwi = (g-n)/(g+n+1e-10)
    nbr  = (n-s)/(n+s+1e-10)
    return ndvi, bsi, ndwi, nbr

def norm(arr):
    arr = np.nan_to_num(arr, nan=0.0)
    arr = np.clip(arr, 0, None)
    return arr/arr.max() if arr.max()>0 else arr

def add_cbar(im, ax, label=""):
    cb = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(label, color="#8b949e", fontsize=8)
    cb.ax.yaxis.set_tick_params(color="#8b949e")
    plt.setp(cb.ax.yaxis.get_ticklabels(), color="#8b949e")

# ── Load data ──────────────────────────────────────────────────────────────
with st.spinner("Loading satellite data..."):
    all_stacks = get_all_data()

s19 = all_stacks[2019]
s23 = all_stacks[2023]

if s19 is None or s23 is None:
    st.error("Could not load satellite data.")
    st.stop()

ndvi_b, bsi_b, ndwi_b, nbr_b = compute_indices(s19)
ndvi_a, bsi_a, ndwi_a, nbr_a = compute_indices(s23)

d_ndvi = norm(ndvi_b - ndvi_a)
d_bsi  = norm(bsi_a  - bsi_b)
d_ndwi = norm(ndwi_b - ndwi_a)
d_nbr  = norm(nbr_b  - nbr_a)

mining_score = 0.35*d_ndvi + 0.30*d_bsi + 0.20*d_ndwi + 0.15*d_nbr
mining_mask  = mining_score > 0.5
affected_ha  = mining_mask.sum() * PX_HA
peak_score   = mining_score.max()

# ── Metrics ────────────────────────────────────────────────────────────────
st.markdown("---")
c1,c2,c3,c4 = st.columns(4)
c1.metric("Affected area",  f"{affected_ha:.0f} ha")
c2.metric("Peak score",     f"{peak_score:.3f}")
c3.metric("Period",         "2019 – 2023")
c4.metric("Resolution",     "60 m")
st.markdown("---")

# ── Tabs ───────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5 = st.tabs([
    "Multi-Index Fusion",
    "Probability Map",
    "Temporal Analysis",
    "Land Cover",
    "False Positive Filter",
])

# ── Tab 1 ──────────────────────────────────────────────────────────────────
with t1:
    st.markdown("### Multi-index fusion")
    st.caption("Four spectral indices combined into one mining probability score.")
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    fig.patch.set_facecolor(BG)
    for ax,(data,cm,title) in zip(axes,[
        (d_ndvi,       "RdYlGn_r", "NDVI change\nvegetation loss (35%)"),
        (d_bsi,        "OrRd",     "BSI change\nbare soil gain (30%)"),
        (d_ndwi,       "RdBu",     "NDWI change\nwater turbidity (20%)"),
        (d_nbr,        "YlOrRd",   "NBR change\nland disturbance (15%)"),
        (mining_score, "hot",      f"Fused score\n{affected_ha:.0f} ha flagged"),
    ]):
        im = ax.imshow(data, cmap=cm)
        ax.set_title(title, color="#c9d1d9", fontsize=8)
        ax.axis("off")
        add_cbar(im, ax)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    w1,w2,w3,w4 = st.columns(4)
    w1.metric("NDVI weight","35%")
    w2.metric("BSI weight", "30%")
    w3.metric("NDWI weight","20%")
    w4.metric("NBR weight", "15%")

# ── Tab 2 ──────────────────────────────────────────────────────────────────
with t2:
    st.markdown("### Mining probability map")
    st.caption("Contour lines at 35%, 50%, and 65% confidence thresholds.")
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.imshow(ndvi_a, cmap="Greys_r", alpha=0.35, vmin=0, vmax=1)
    sd = np.ma.masked_where(mining_score < 0.2, mining_score)
    im = ax.imshow(sd, cmap="YlOrRd", vmin=0.2, vmax=1.0, alpha=0.9)
    ct = ax.contour(mining_score, levels=[0.35,0.50,0.65],
                    colors=["#ffff00","#ff8800","#ff0000"],
                    linewidths=[0.8,1.2,1.6])
    ax.clabel(ct, fmt={0.35:"35%",0.50:"50%",0.65:"65%"},
              colors="white", fontsize=8)
    add_cbar(im, ax, "Mining probability")
    affected = (mining_score>0.35).sum()*PX_HA
    critical  = (mining_score>0.65).sum()*PX_HA
    ax.text(0.02,0.02,
            f"Affected (>35%) : {affected:.0f} ha\nCritical (>65%) : {critical:.0f} ha\nPeak score      : {peak_score:.3f}",
            transform=ax.transAxes, color="white", fontsize=9,
            va="bottom", family="monospace",
            bbox=dict(facecolor=BG, edgecolor="#30363d", alpha=0.9, boxstyle="round,pad=0.4"))
    ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    ca,cb = st.columns(2)
    ca.metric("Area above 35%", f"{affected:.0f} ha")
    cb.metric("Critical above 65%", f"{critical:.0f} ha")

# ── Tab 3 ──────────────────────────────────────────────────────────────────
with t3:
    st.markdown("### Temporal anomaly detection")
    st.caption("Year-on-year mining score and vegetation trend 2019–2023.")
    yearly_scores, yearly_ndvi, loaded_years = [], [], []
    for year in YEARS:
        s = all_stacks.get(year)
        if s is None: continue
        n,b,w_,r = compute_indices(s)
        sc = 0.35*norm(n)+0.30*norm(b)+0.20*norm(w_)+0.15*norm(r)
        yearly_scores.append(float(np.nanmean(sc)))
        yearly_ndvi.append(float(np.nanmean(np.nan_to_num(n))))
        loaded_years.append(year)
    scores = np.array(yearly_scores)
    mean_s = scores.mean(); std_s = scores.std(); thresh = mean_s+std_s
    pt_colors = ["#d62728" if s>mean_s+2*std_s else "#ff7f0e" if s>thresh else "#2ca02c" for s in scores]
    fig, axes = plt.subplots(1,2, figsize=(16,5))
    fig.patch.set_facecolor(BG)
    ax = axes[0]
    ax.fill_between(loaded_years, mean_s-std_s, mean_s+std_s, alpha=0.15, color="#2ca02c", label="Normal range")
    ax.axhline(thresh, color="#d62728", linestyle="--", linewidth=1.2, label=f"Alert ({thresh:.4f})")
    ax.plot(loaded_years, scores, color="#58a6ff", linewidth=2, marker="o", markersize=8, zorder=3)
    for yr,s,c in zip(loaded_years,scores,pt_colors):
        ax.scatter(yr,s,color=c,s=120,zorder=5)
        ax.annotate(f"{s:.4f}",(yr,s),textcoords="offset points",xytext=(0,10),fontsize=8,ha="center",color="#c9d1d9")
    ax.set_xticks(loaded_years); ax.set_xlabel("Year"); ax.set_ylabel("Mean mining score")
    ax.set_title("Mining score trend", color="#c9d1d9"); ax.legend(fontsize=8); ax.grid(alpha=0.15)
    ax2 = axes[1]
    ax2.bar(loaded_years, yearly_ndvi, color="#4c72b0", width=0.6, edgecolor="#30363d")
    ax2.set_xticks(loaded_years); ax2.set_xlabel("Year"); ax2.set_ylabel("Mean NDVI")
    ax2.set_title("Vegetation health trend", color="#c9d1d9"); ax2.grid(alpha=0.15, axis="y")
    for yr,v in zip(loaded_years,yearly_ndvi):
        ax2.text(yr, v+0.002, f"{v:.3f}", ha="center", fontsize=8, color="#c9d1d9")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    st.markdown("**Year-by-year summary**")
    for yr,s in zip(loaded_years,scores):
        status = "Critical" if s>mean_s+2*std_s else "Suspected" if s>thresh else "Normal"
        st.write(f"{yr} — Score: `{s:.4f}` — {status}")

# ── Tab 4 ──────────────────────────────────────────────────────────────────
with t4:
    st.markdown("### Land cover classification")
    st.caption("K-Means clustering on four spectral indices. Forest, water, degraded, mining.")
    h,w = ndvi_a.shape
    fa = np.stack([np.nan_to_num(ndvi_a),np.nan_to_num(bsi_a),
                   np.nan_to_num(ndwi_a),np.nan_to_num(nbr_a)],axis=-1).reshape(-1,4)
    fb = np.stack([np.nan_to_num(ndvi_b),np.nan_to_num(bsi_b),
                   np.nan_to_num(ndwi_b),np.nan_to_num(nbr_b)],axis=-1).reshape(-1,4)
    km = KMeans(n_clusters=4,random_state=42,n_init=10)
    km.fit(fa)
    la = km.predict(fa).reshape(h,w)
    lb = km.predict(fb).reshape(h,w)
    centers = km.cluster_centers_
    fc = int(np.argmax(centers[:,0]))
    mc = int(np.argmax(centers[:,1]))
    wc = int(np.argmax(centers[:,2]))
    dc = list({0,1,2,3}-{fc,mc,wc})[0]
    ci = {fc:("Forest / Vegetation","#2d8a2d"),wc:("Water bodies","#3399ff"),
          dc:("Degraded land","#ff8800"),mc:("Active mining","#ff2222")}
    rf = {fc:0,wc:1,dc:2,mc:3}
    rfn = np.vectorize(rf.get)
    cmap_lc = ListedColormap(["#2d8a2d","#3399ff","#ff8800","#ff2222"])
    fig,axes = plt.subplots(1,2,figsize=(16,6))
    fig.patch.set_facecolor(BG)
    for ax,disp,year in zip(axes,[rfn(lb),rfn(la)],["2019","2023"]):
        ax.imshow(disp,cmap=cmap_lc,vmin=0,vmax=3,interpolation="nearest")
        ax.set_title(f"Land cover {year}",color="#c9d1d9",fontsize=12)
        ax.axis("off")
    patches = [mpatches.Patch(color=c,label=n) for _,(n,c) in ci.items()]
    axes[1].legend(handles=patches,loc="lower left",fontsize=9)
    m19=(lb==mc).sum()*PX_HA; m23=(la==mc).sum()*PX_HA
    f19=(lb==fc).sum()*PX_HA; f23=(la==fc).sum()*PX_HA
    axes[1].text(0.98,0.98,
        f"Mining\n  2019: {m19:.0f} ha\n  2023: {m23:.0f} ha\n  +{m23-m19:.0f} ha\n\nForest\n  2019: {f19:.0f} ha\n  2023: {f23:.0f} ha\n  {f23-f19:.0f} ha",
        transform=axes[1].transAxes,color="white",fontsize=9,va="top",ha="right",
        family="monospace",bbox=dict(facecolor=BG,edgecolor="#ff2222",alpha=0.9,boxstyle="round,pad=0.4"))
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    ca,cb,cc,cd = st.columns(4)
    ca.metric("Mining 2019", f"{m19:.0f} ha")
    cb.metric("Mining 2023", f"{m23:.0f} ha", delta=f"+{m23-m19:.0f} ha")
    cc.metric("Forest 2019", f"{f19:.0f} ha")
    cd.metric("Forest 2023", f"{f23:.0f} ha", delta=f"{f23-f19:.0f} ha")

# ── Tab 5 ──────────────────────────────────────────────────────────────────
with t5:
    st.markdown("### False positive filtering")
    st.caption("Urban and agricultural areas removed using OpenStreetMap landuse data.")
    with st.spinner("Fetching OSM landuse data..."):
        try:
            resp = requests.get("http://overpass-api.de/api/interpreter", params={"data":"""
[out:json][timeout:25];
(way["landuse"~"residential|farmland|farm|industrial"](23.5,85.8,23.8,86.2);
 way["place"~"village|town|city|hamlet"](23.5,85.8,23.8,86.2););
out geom;"""}, timeout=30)
            elements = resp.json().get("elements",[])
            h_,w_ = mining_score.shape
            fp = np.zeros((h_,w_),dtype=bool)
            for el in elements:
                if "geometry" not in el: continue
                for n in el["geometry"]:
                    col=int((n["lon"]-85.8)/(86.2-85.8)*w_)
                    row=int((23.8-n["lat"])/(23.8-23.5)*h_)
                    if 0<=row<h_ and 0<=col<w_: fp[row,col]=True
            sf = mining_score.copy(); sf[fp]=0
        except:
            sf = mining_score.copy()
            st.warning("OSM unavailable. Showing unfiltered map.")
    bh=(mining_score>0.35).sum()*PX_HA
    ah=(sf>0.35).sum()*PX_HA
    rh=bh-ah
    fig,axes = plt.subplots(1,2,figsize=(16,6))
    fig.patch.set_facecolor(BG)
    for ax,score,title in zip(axes,[mining_score,sf],
        [f"Before filter\n{bh:.0f} ha flagged",f"After filter\n{ah:.0f} ha ({rh:.0f} ha removed)"]):
        ax.imshow(ndvi_a,cmap="Greys_r",alpha=0.35,vmin=0,vmax=1)
        sd=np.ma.masked_where(score<0.2,score)
        im=ax.imshow(sd,cmap="YlOrRd",vmin=0.2,vmax=1.0,alpha=0.9)
        ax.set_title(title,color="#c9d1d9",fontsize=11); ax.axis("off")
    add_cbar(im,axes[1],"Mining probability")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    ca,cb = st.columns(2)
    ca.metric("Before filter", f"{bh:.0f} ha")
    cb.metric("After filter",  f"{ah:.0f} ha", delta=f"-{rh:.0f} ha")
