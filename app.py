"""
MRI Phantom Simulator (SDF)
---------------------------------

A Streamlit app that:
- Accepts a **2D or 3D SDF** file (only) for the test molecule.
- Simulates paramagnetic effects across **8 circular phantoms** arranged **2 columns x 4 rows**:
  
    Row 1: 1/4 | 1/8
    Row 2: 1/2 | 1/16
    Row 3: 1   | 1/32
    Row 4: Gd  | Saf Su

- Provides a UI to:
  - Choose **TR** (single selection from a customizable list) while keeping **TE constant**, **or** choose a **TE** (single selection from a customizable list) while keeping **TR constant**.
  - Enter relaxivities **r1, r2** (s⁻¹·mM⁻¹) for both the **test molecule** and **Gd reference**.
  - Enter **stock concentration** for the test molecule (mM) to auto-generate dilutions: 1, 1/2, 1/4, 1/8, 1/16, 1/32.
  - Enter baseline **T1₀, T2₀** for water (ms).
  - Add optional Gaussian noise.
- Renders a synthetic MR magnitude image (spin-echo model):
    S ∝ PD · (1 − e^(−TR/T1)) · e^(−TE/T2)
- Reports **ROI mean values** (per phantom) and allows CSV export.
- Optionally draws the uploaded molecule if **RDKit** is installed.

Notes
-----
- The simulation is **not intended to be 100% accurate**; it aims to be internally consistent and adjustable.
- Units: TR, TE, T1, T2 in **ms**; concentrations in **mM**; relaxivities r1, r2 in **s⁻¹·mM⁻¹**.
- Requires Python 3.9+ recommended.

Suggested install:
    pip install streamlit numpy matplotlib pandas pillow
    # Optional for SDF parsing + drawing:
    pip install rdkit-pypi

Run:
    streamlit run app.py
"""

from __future__ import annotations
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# --- Optional RDKit support (for SDF parsing / 2D depiction) ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.Draw import MolToImage
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

import streamlit as st

# ------------------------------
# Data classes & helpers
# ------------------------------
@dataclass
class Relaxivity:
    r1: float  # [s^-1 mM^-1]
    r2: float  # [s^-1 mM^-1]

@dataclass
class Baseline:
    T10_ms: float
    T20_ms: float
    PD: float = 1.0  # relative proton density

@dataclass
class Species:
    name: str
    relax: Relaxivity
    conc_mM: float  # concentration (mM)


def parse_sdf(file_bytes: bytes) -> Dict[str, Optional[str]]:
    """Parse SDF to extract basic metadata and (optionally) a depiction.
    Returns a dict with title, formula, exactmw, and PIL image if RDKit is available.
    """
    info = {"title": None, "formula": None, "mw": None, "image": None}

    if not RDKit_AVAILABLE:
        # Minimal: we can only expose the first line as title (typical SDF style)
        try:
            head = file_bytes.splitlines()[0].decode(errors="ignore").strip()
            info["title"] = head if head else "(SDF yüklendi)"
        except Exception:
            info["title"] = "(SDF yüklendi)"
        return info

    suppl = Chem.SDMolSupplier()
    # SDMolSupplier expects a file-like; use BytesIO temp file workaround
    tmp = io.BytesIO(file_bytes)
    # rdkit needs a file path or file-like with name; fallback to MolFromMolBlock
    try:
        sdf_text = file_bytes.decode("utf-8", errors="ignore")
        mol = Chem.MolFromMolBlock(sdf_text, sanitize=True, removeHs=False)
    except Exception:
        mol = None

    if mol is None:
        # Try supplier route
        try:
            with Chem.SDMolSupplier() as _:
                pass
        except Exception:
            pass
        # Last attempt
        try:
            mol = Chem.MolFromMolBlock(file_bytes.decode(errors="ignore"))
        except Exception:
            mol = None

    if mol is None:
        # Fallback: just capture header
        try:
            head = file_bytes.splitlines()[0].decode(errors="ignore").strip()
            info["title"] = head if head else "(SDF yüklendi)"
        except Exception:
            info["title"] = "(SDF yüklendi)"
        return info

    try:
        info["title"] = mol.GetProp("_Name") if mol.HasProp("_Name") else "(Molekül)"
    except Exception:
        info["title"] = "(Molekül)"

    try:
        info["mw"] = f"{Descriptors.MolWt(mol):.2f}"
    except Exception:
        info["mw"] = None

    try:
        # Empirical formula is not direct; we can compute via atom counts
        formula_dict = {}
        for atom in mol.GetAtoms():
            sym = atom.GetSymbol()
            formula_dict[sym] = formula_dict.get(sym, 0) + 1
        # Order: C, H, then alphabetical
        def formula_key(k):
            if k == "C":
                return (0, k)
            if k == "H":
                return (1, k)
            return (2, k)
        parts = [f"{el}{(n if n>1 else '')}" for el, n in sorted(formula_dict.items(), key=lambda kv: formula_key(kv[0]))]
        info["formula"] = "".join(parts)
    except Exception:
        info["formula"] = None

    try:
        img = MolToImage(mol, size=(280, 220))
        info["image"] = img
    except Exception:
        info["image"] = None

    return info


# ------------------------------
# Physics / signal model
# ------------------------------

def apply_relaxivity(T10_ms: float, T20_ms: float, r1: float, r2: float, conc_mM: float) -> Tuple[float, float]:
    """Compute T1 and T2 (ms) from baseline T10, T20 and relaxivities at given concentration.
    1/T1 = 1/T10 + r1*C   and   1/T2 = 1/T20 + r2*C, (r in s^-1·mM^-1, C in mM)
    """
    # convert ms -> s
    T10 = max(T10_ms, 1e-6) / 1000.0
    T20 = max(T20_ms, 1e-6) / 1000.0

    R1 = 1.0 / T10 + r1 * conc_mM
    R2 = 1.0 / T20 + r2 * conc_mM

    T1_s = 1.0 / max(R1, 1e-9)
    T2_s = 1.0 / max(R2, 1e-9)

    return T1_s * 1000.0, T2_s * 1000.0  # return ms


def spin_echo_signal(TR_ms: float, TE_ms: float, T1_ms: float, T2_ms: float, PD: float = 1.0) -> float:
    TR = max(TR_ms, 1e-9)
    TE = max(TE_ms, 1e-9)
    T1 = max(T1_ms, 1e-9)
    T2 = max(T2_ms, 1e-9)
    return float(PD * (1.0 - math.exp(-TR / T1)) * math.exp(-TE / T2))


# ------------------------------
# Phantom layout & rendering
# ------------------------------

Position = Tuple[int, int]

LAYOUT = [
    ("1/4",  0.25), ("1/8",  0.125),
    ("1/2",  0.50), ("1/16", 0.0625),
    ("1",    1.0),  ("1/32", 0.03125),
    ("Gd",   "gd"), ("Saf Su", "water"),
]


def make_circle_mask(h: int, w: int, center: Position, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = center
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return dist2 <= radius ** 2


def render_phantom(signals: Dict[str, float], size: Tuple[int, int] = (900, 1200), noise_sigma: float = 0.0,
                   annotate: bool = True, normalize: bool = True) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Position], int]:
    """Render the 2x4 circular phantom grid to a 2D image array in [0,1].
    Returns (image, masks, centers, radius)
    """
    h, w = size
    img = np.zeros((h, w), dtype=np.float32)

    # Grid geometry
    rows, cols = 4, 2
    pad_y, pad_x = 40, 40
    cell_h = (h - (rows + 1) * pad_y) // rows
    cell_w = (w - (cols + 1) * pad_x) // cols
    radius = int(min(cell_h, cell_w) * 0.35)

    centers: Dict[str, Position] = {}
    masks: Dict[str, np.ndarray] = {}

    # Order is fixed by LAYOUT
    for idx, (label, _) in enumerate(LAYOUT):
        r = idx // 2
        c = idx % 2
        cy = pad_y + r * (cell_h + pad_y) + cell_h // 2
        cx = pad_x + c * (cell_w + pad_x) + cell_w // 2
        centers[label] = (cy, cx)
        mask = make_circle_mask(h, w, (cy, cx), radius)
        masks[label] = mask
        val = float(max(signals.get(label, 0.0), 0.0))
        img[mask] = val

    # Optional normalization across phantoms
    if normalize:
        m = float(img.max())
        if m > 1e-9:
            img = img / m
    # Add Gaussian noise (magnitude image proxy)
    if noise_sigma > 0:
        noise = np.random.normal(loc=0.0, scale=noise_sigma, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    # Simple vignette / background bias to look MRI-ish
    Y, X = np.indices(img.shape)
    Y = (Y - h / 2) / (h / 2)
    X = (X - w / 2) / (w / 2)
    rad = np.sqrt(X**2 + Y**2)
    vignette = np.clip(1.0 - 0.1 * rad, 0.9, 1.0)
    img = np.clip(img * vignette, 0.0, 1.0)

    return img, masks, centers, radius


def compute_rois(img: np.ndarray, masks: Dict[str, np.ndarray]) -> Dict[str, float]:
    rois = {}
    for label, m in masks.items():
        if m.sum() > 0:
            rois[label] = float(img[m].mean())
        else:
            rois[label] = float("nan")
    return rois


# ------------------------------
# Streamlit UI
# ------------------------------

st.set_page_config(page_title="MRI Phantom Simulator (SDF) by Alperen Can Esen", layout="wide")

st.title("MRI Phantom Simulator (SDF) by Alperen Can Esen")
st.caption("Paramanyetik etkileri 8 fantom üzerinde görselleştir ve ROI ortalamalarını hesapla. (Deneysel/öğretici simülasyon)")

# Sidebar: Upload SDF
with st.sidebar:
    st.header("Molekül (SDF)")
    sdf_file = st.file_uploader("Sadece .sdf dosyası yükleyin", type=["sdf"], accept_multiple_files=False)
    mol_info = None
    if sdf_file is not None:
        sdf_bytes = sdf_file.read()
        mol_info = parse_sdf(sdf_bytes)
        st.success("SDF yüklendi.")
        if mol_info.get("image") is not None:
            st.image(mol_info["image"], caption=mol_info.get("title") or "Molekül", use_column_width=True)
        else:
            st.write(mol_info.get("title") or "Molekül")
            if mol_info.get("formula"):
                st.write(f"Formül: {mol_info['formula']}")
            if mol_info.get("mw"):
                st.write(f"Mol kütlesi: {mol_info['mw']} g/mol")
    else:
        st.info("Başlamak için 2D/3D bir **SDF** dosyası yükleyin.")

    st.divider()
    st.header("Model Parametreleri")
    col_a, col_b = st.columns(2)
    with col_a:
        T10_ms = st.number_input("Saf su T1₀ (ms)", min_value=100.0, max_value=10000.0, value=3000.0, step=50.0)
        T20_ms = st.number_input("Saf su T2₀ (ms)", min_value=10.0, max_value=5000.0, value=2000.0, step=10.0)
        PD = st.slider("Proton yoğunluğu (göreli)", min_value=0.1, max_value=2.0, value=1.0, step=0.1)
        noise_sigma = st.slider("Gürültü (σ, 0-0.1)", min_value=0.0, max_value=0.1, value=0.01, step=0.005)
        normalize = st.checkbox("Görüntüyü normalize et (0-1)", value=True)
    with col_b:
        st.subheader("Test molekül (SDF)")
        stock_mM = st.number_input("Stok derişim (mM)", min_value=0.0, max_value=1000.0, value=1.0, step=0.1)
        r1_test = st.number_input("r1 (s⁻¹·mM⁻¹) — test", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
        r2_test = st.number_input("r2 (s⁻¹·mM⁻¹) — test", min_value=0.0, max_value=50.0, value=1.2, step=0.1)
        st.subheader("Gd referansı")
        gd_mM = st.number_input("Gd derişim (mM)", min_value=0.0, max_value=50.0, value=1.0, step=0.1)
        r1_gd = st.number_input("r1 (s⁻¹·mM⁻¹) — Gd", min_value=0.0, max_value=50.0, value=4.5, step=0.1)
        r2_gd = st.number_input("r2 (s⁻¹·mM⁻¹) — Gd", min_value=0.0, max_value=50.0, value=5.0, step=0.1)

    st.divider()
    st.header("Zamanlama (TR/TE)")
    mode = st.radio("Simülasyon modu", ["TR taraması (TE sabit)", "TE taraması (TR sabit)"], index=0)

    if mode.startswith("TR"):
        TR_list_str = st.text_input("TR listesi (ms, virgülle)", value="75,150,300,600,1200,2400,4800")
        try:
            TR_list = [float(x) for x in TR_list_str.replace(";", ",").split(",") if x.strip()]
        except Exception:
            TR_list = [75,150,300,600,1200,2400,4800]
        TR_sel = st.select_slider("Seçili TR (ms)", options=TR_list, value=TR_list[min(2, len(TR_list)-1)])
        TE_fixed = st.number_input("TE (ms) — sabit", min_value=1.0, max_value=1000.0, value=80.0, step=1.0)
    else:
        TE_list_str = st.text_input("TE listesi (ms, virgülle)", value="10,20,40,80,120")
        try:
            TE_list = [float(x) for x in TE_list_str.replace(";", ",").split(",") if x.strip()]
        except Exception:
            TE_list = [10,20,40,80,120]
        TE_sel = st.select_slider("Seçili TE (ms)", options=TE_list, value=TE_list[min(2, len(TE_list)-1)])
        TR_fixed = st.number_input("TR (ms) — sabit", min_value=1.0, max_value=10000.0, value=600.0, step=10.0)


# Main area
left, right = st.columns([0.6, 0.4])

with left:
    st.subheader("Fantom görüntüsü (2 sütun × 4 satır)")

    if sdf_file is None:
        st.warning("Simülasyon için önce bir **SDF** dosyası yükleyin.")

    # Prepare species/phantoms
    base = Baseline(T10_ms=T10_ms, T20_ms=T20_ms, PD=PD)
    test_relax = Relaxivity(r1=r1_test, r2=r2_test)
    gd_relax = Relaxivity(r1=r1_gd, r2=r2_gd)

    # Concentrations per site (mM)
    conc_map: Dict[str, float] = {
        "1":      stock_mM * 1.0,
        "1/2":    stock_mM * 0.5,
        "1/4":    stock_mM * 0.25,
        "1/8":    stock_mM * 0.125,
        "1/16":   stock_mM * 0.0625,
        "1/32":   stock_mM * 0.03125,
        "Gd":     gd_mM,
        "Saf Su": 0.0,
    }

    # Compute T1/T2 and signal per phantom
    def compute_signals_for(TR_ms: float, TE_ms: float) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        signals: Dict[str, float] = {}
        rows: List[Dict[str, float]] = []
        for label, frac in LAYOUT:
            if label == "Gd":
                T1_ms, T2_ms = apply_relaxivity(base.T10_ms, base.T20_ms, gd_relax.r1, gd_relax.r2, conc_map[label])
            elif label == "Saf Su":
                T1_ms, T2_ms = base.T10_ms, base.T20_ms
            else:
                c_mM = conc_map[label if isinstance(frac, float) else label]
                T1_ms, T2_ms = apply_relaxivity(base.T10_ms, base.T20_ms, test_relax.r1, test_relax.r2, c_mM)

            S = spin_echo_signal(TR_ms, TE_ms, T1_ms, T2_ms, PD=base.PD)
            signals[label] = S
            rows.append({
                "Phantom": label,
                "Kons. (mM)": conc_map[label] if label in conc_map else float("nan"),
                "T1 (ms)": T1_ms,
                "T2 (ms)": T2_ms,
                "Sinyal (a.u.)": S,
            })
        return signals, rows

    if mode.startswith("TR"):
        TR_curr, TE_curr = TR_sel, TE_fixed
    else:
        TR_curr, TE_curr = TR_fixed, TE_sel

    signals, rows = compute_signals_for(TR_curr, TE_curr)

    # Render image
    img, masks, centers, radius = render_phantom(signals, size=(900, 1200), noise_sigma=noise_sigma, normalize=normalize)

    # Draw with annotations
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")

    # Annotate labels and ROI means (from clean signal, not noisy image)
    for label, (cy, cx) in centers.items():
        ax.text(cx, cy, label, ha="center", va="center", fontsize=12, color="w")

    st.pyplot(fig, use_container_width=True)

    # Export PNG
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    st.download_button("Görüntüyü PNG olarak indir", data=buf, file_name="phantom_sim.png", mime="image/png")

with right:
    st.subheader("ROI & Parametre Özeti")

    df = pd.DataFrame(rows)
    # Sort for readability in the specified grid order
    order = [lbl for lbl, _ in LAYOUT]
    df["_ord"] = df["Phantom"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)

    st.dataframe(df, use_container_width=True, hide_index=True)

    # CSV export
    csv = df.to_csv(index=False).encode("utf-8")
    fname = f"roi_TR{int(TR_curr)}_TE{int(TE_curr)}.csv"
    st.download_button("ROI değerlerini CSV indir", data=csv, file_name=fname, mime="text/csv")

    st.divider()
    st.subheader("Seçenekler")
    if mode.startswith("TR"):
        st.markdown(f"**TR** = `{TR_curr} ms`, **TE** = `{TE_curr} ms` (sabit)")
    else:
        st.markdown(f"**TR** = `{TR_curr} ms` (sabit), **TE** = `{TE_curr} ms`")

    st.caption("Not: Bu simülasyon öğretici amaçlıdır ve mutlak doğruluk hedeflenmemiştir. Parametreleri değiştirerek göreli etkileri inceleyebilirsiniz. Hesaplamalar tamamen sizin verdiğiniz parametrelere bağlıdır.")

# Footer context
st.markdown("""
---
**Yerleşim:**

```
1/4 | 1/8
1/2 | 1/16
1   | 1/32
Gd  | Saf Su
```

**Model:** Spin-echo sinyali:  S = PD · (1 − e^(−TR/T1)) · e^(−TE/T2)

**Derişimler:** Test molekülü için otomatik dilüsyonlar (1, 1/2, 1/4, 1/8, 1/16, 1/32). Gd ve Saf Su ayrı referanslardır.
""")
