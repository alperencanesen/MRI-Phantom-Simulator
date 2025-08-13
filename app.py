"""
MRI Phantom Simulator (SDF) — Auto In‑Silico Relaxivity Estimator
=================================================================
Yalnızca **SDF** + **TR/TE** verirsin; program molekülü analiz edip **r1/r2**
relaksivite tahminini **otomatik** çıkarır. Elle radikal/override yok.

Yaklaşım (özet)
- RDKit ile SDF’ten molekül okunur ve şu öznitelikler hesaplanır:
  metal merkez(ler)i, radikal e− sayısı (varsa), konjugasyon/ aromatiklik,
  heteroatom sayısı (N,O,S), halka sayısı, formel yük, vb.
- Bu özniteliklerden bir **paramanyetik eğilim skoru (para_score)** türetilir.
- Sonra sezgisel bir eşleme ile **r1 = a0 + a1·para_score**, **r2 = b0 + b1·para_score**
  hesaplanır (Gd vb. metaller için daha yüksek ağırlıklar).
- SDF property bloklarında sayısal `r1`, `r2` varsa **doğrudan** onlar kullanılır.

Notlar
- Bu bir **in‑silico kestirim**dir; amaç %100 isabet değil, **genelleştirilebilir ve tutarlı** davranış.
- RDKit yoksa temel bir varsayılan (zayıf paramanyetik) kullanılır.
- Baseline su için T1₀/T2₀ sabit alınır (aşağıdaki sabitler bölümünde değiştirilebilir).
"""

from __future__ import annotations
import io
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Optional RDKit support ---
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.Chem.Draw import MolToImage
    RDKit_AVAILABLE = True
except Exception:
    RDKit_AVAILABLE = False

import streamlit as st

# Sayfa ayarı en başta
st.set_page_config(page_title="MRI Phantom Simulator (SDF) — Auto In‑Silico", layout="wide")

# ==============================
# Sabitler (gerekirse düzenleyin)
# ==============================
DEFAULT_T10_MS = 4000.0  # Saf su T1₀ (ms)
DEFAULT_T20_MS = 1160.0  # Saf su T2₀ (ms)
DEFAULT_PD     = 1.0

STOCK_MOLAR_M = 1.0      # Test molekülü stok derişim (mM)
GD_MOLAR_M    = 1.0      # Gd referans derişimi (mM)
GD_R1         = 4.5      # [s^-1 mM^-1]
GD_R2         = 5.0      # [s^-1 mM^-1]

IMG_NOISE_SIGMA = 0.01   # Görsel gürültü
NORMALIZE_IMAGE = True

# Görsel kontrast: gamma<1 -> düşük sinyallerin daha beyaz görünmesi için
DISPLAY_GAMMA = 0.72  # 0.6–0.85 arası daha parlak; 1.0 kapalı gibi davranır

# ==============================
# Veri sınıfları & model
# ==============================
@dataclass
class Relaxivity:
    r1: float  # [s^-1 mM^-1]
    r2: float  # [s^-1 mM^-1]

@dataclass
class Baseline:
    T10_ms: float
    T20_ms: float
    PD: float = 1.0

Position = Tuple[int, int]

LAYOUT = [
    ("1/4",  0.25), ("1/8",  0.125),
    ("1/2",  0.50), ("1/16", 0.0625),
    ("1",    1.0),  ("1/32", 0.03125),
    ("Gd",   "gd"), ("Saf Su", "water"),
]

# ------------------------------
# SDF okuma
# ------------------------------

def parse_sdf(file_bytes: bytes) -> Dict[str, Optional[str]]:
    info = {"title": None, "formula": None, "mw": None, "image": None, "mol": None, "props": {}, "sdf_text": None}
    if not RDKit_AVAILABLE:
        try:
            head = file_bytes.splitlines()[0].decode(errors="ignore").strip()
            info["title"] = head if head else "(SDF yüklendi)"
        except Exception:
            info["title"] = "(SDF yüklendi)"
        return info
    try:
        sdf_text = file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        sdf_text = None
    info["sdf_text"] = sdf_text
    mol = None
    if sdf_text:
        try:
            mol = Chem.MolFromMolBlock(sdf_text, sanitize=True, removeHs=False)
        except Exception:
            mol = None
    if mol is None:
        try:
            mol = Chem.MolFromMolBlock(file_bytes.decode(errors="ignore"))
        except Exception:
            mol = None
    if mol is None:
        try:
            head = file_bytes.splitlines()[0].decode(errors="ignore").strip()
            info["title"] = head if head else "(SDF yüklendi)"
        except Exception:
            info["title"] = "(SDF yüklendi)"
        return info
    info["mol"] = mol
    try:
        info["title"] = mol.GetProp("_Name") if mol.HasProp("_Name") else "(Molekül)"
    except Exception:
        info["title"] = "(Molekül)"
    try:
        info["mw"] = f"{Descriptors.MolWt(mol):.2f}"
    except Exception:
        info["mw"] = None
    try:
        counts = {}
        for a in mol.GetAtoms():
            s = a.GetSymbol()
            counts[s] = counts.get(s, 0) + 1
        def key(k):
            return (0, k) if k == "C" else (1, k) if k == "H" else (2, k)
        info["formula"] = "".join(f"{el}{(n if n>1 else '')}" for el, n in sorted(counts.items(), key=lambda kv: key(kv[0])))
    except Exception:
        info["formula"] = None
    try:
        info["image"] = MolToImage(mol, size=(280, 220))
    except Exception:
        info["image"] = None
    try:
        info["props"] = {k.lower(): v for k, v in mol.GetPropsAsDict().items()}
    except Exception:
        info["props"] = {}
    return info

# ------------------------------
# Basit SDF özellik çıkarıcı (RDKit yoksa)
# ------------------------------

PT_Z = {
    'H':1,'He':2,'Li':3,'Be':4,'B':5,'C':6,'N':7,'O':8,'F':9,'Ne':10,
    'Na':11,'Mg':12,'Al':13,'Si':14,'P':15,'S':16,'Cl':17,'Ar':18,'K':19,'Ca':20,
    'Sc':21,'Ti':22,'V':23,'Cr':24,'Mn':25,'Fe':26,'Co':27,'Ni':28,'Cu':29,'Zn':30,
    'Ga':31,'Ge':32,'As':33,'Se':34,'Br':35,'Kr':36,'Rb':37,'Sr':38,'Y':39,'Zr':40,
    'Nb':41,'Mo':42,'Tc':43,'Ru':44,'Rh':45,'Pd':46,'Ag':47,'Cd':48,'In':49,'Sn':50,
    'Sb':51,'Te':52,'I':53,'Xe':54,'Cs':55,'Ba':56,'La':57,'Ce':58,'Pr':59,'Nd':60,
    'Pm':61,'Sm':62,'Eu':63,'Gd':64,'Tb':65,'Dy':66,'Ho':67,'Er':68,'Tm':69,'Yb':70,'Lu':71,
    'Hf':72,'Ta':73,'W':74,'Re':75,'Os':76,'Ir':77,'Pt':78,'Au':79,'Hg':80,'Tl':81,'Pb':82,
    'Bi':83,'Po':84,'At':85,'Rn':86,'Fr':87,'Ra':88,'Ac':89,'Th':90,'Pa':91,'U':92
}

def parse_sdf_basic_features(sdf_text: Optional[str]) -> Dict[str, float]:
    # Varsayılan boş özellik seti
    feats = {"radicals":0.0,"metal":0.0,"conj":0.0,"arom_frac":0.0,
             "hetero":0.0,"rings":0.0,"charge":0.0,"heavy":0.0}
    if not sdf_text:
        return feats
    lines = [ln.rstrip() for ln in sdf_text.splitlines()]
    # Counts satırını bul (V2000)
    counts_idx = None
    for i,ln in enumerate(lines[:10]):
        if ln.strip().endswith("V2000") and len(ln) >= 6:
            counts_idx = i
            break
    if counts_idx is None:
        # V3000 veya farklı ise minimum çıkarımlar: hetero/metaller property aramayla yok
        return feats
    try:
        ln = lines[counts_idx]
        nat = int(ln[0:3])
        nbo = int(ln[3:6])
    except Exception:
        return feats
    # Atom ve bağ kayıtlarını oku
    atoms = []
    for j in range(nat):
        s = lines[counts_idx+1+j]
        toks = s.split()
        sym = toks[3] if len(toks) >= 4 else s[31:34].strip()
        sym = sym.strip()
        z = PT_Z.get(sym, 0)
        atoms.append((sym, z))
    bonds = []
    for j in range(nbo):
        s = lines[counts_idx+1+nat+j]
        toks = s.split()
        if len(toks) >= 3:
            try:
                a1 = int(toks[0]); a2 = int(toks[1]); btyp = int(toks[2]) if len(toks) > 2 else 1
            except Exception:
                continue
            bonds.append((a1-1, a2-1, btyp))
    # Bileşen sayısı (Union-Find)
    parent = list(range(len(atoms)))
    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb = find(a),find(b)
        if ra!=rb: parent[rb]=ra
    for a1,a2,_ in bonds:
        if 0<=a1<len(atoms) and 0<=a2<len(atoms):
            union(a1,a2)
    comps = len({find(i) for i in range(len(atoms))}) if atoms else 0
    V = len(atoms); E = len(bonds)
    rings = max(0, E - V + comps)  # siklomatik sayı ≈ halka sayısı

    # Hetero, heavy, metal skoru
    hetero = sum(1 for sym,z in atoms if z in (7,8,16))
    heavy = sum(1 for sym,z in atoms if z > 1)
    # metal skoru
    atomic_nums = [z for _,z in atoms]
    mscore = 0.0
    for z in atomic_nums:
        if z in TRANSITION_Z or z in LANTHANIDES_Z or z in ACTINIDES_Z:
            mscore += METAL_WEIGHTS.get(z, DEFAULT_DF_WEIGHT)

    # Konjugasyon ve aromatiklik yaklaşımı
    conj_bonds = sum(1 for _,_,t in bonds if t in (2,4))
    conj = (conj_bonds / max(1,E))
    arom_atoms = set()
    for a1,a2,t in bonds:
        if t == 4:
            arom_atoms.add(a1); arom_atoms.add(a2)
    arom_frac = (len(arom_atoms) / max(1, heavy))

    # Radikal ve yük (M  RAD / M  CHG satırlarından)
    radicals = 0
    charge = 0
    for ln in lines:
        if ln.startswith("M  RAD"):
            toks = ln.split()
            # format: M RAD n  idx1 rad1 idx2 rad2 ...
            try:
                pairs = toks[3:]
                for k in range(0,len(pairs),2):
                    rad = int(pairs[k+1])
                    radicals += rad
            except Exception:
                pass
        elif ln.startswith("M  CHG"):
            toks = ln.split()
            try:
                pairs = toks[3:]
                for k in range(0,len(pairs),2):
                    ch = int(pairs[k+1])
                    charge += abs(ch)
            except Exception:
                pass
    feats.update({
        "radicals": float(radicals),
        "metal": float(mscore),
        "conj": float(conj),
        "arom_frac": float(arom_frac),
        "hetero": float(hetero),
        "rings": float(rings),
        "charge": float(charge),
        "heavy": float(heavy),
    })
    return feats

# ------------------------------
# In‑silico r1/r2 tahminleyici (heuristic)
# ------------------------------
TRANSITION_Z = set(range(21, 31)) | set(range(39, 49))
LANTHANIDES_Z = set(range(57, 72))
ACTINIDES_Z = set(range(89, 104))

METAL_WEIGHTS = {
    24: 2.0, 25: 3.5, 26: 3.0, 27: 2.5, 28: 2.0, 29: 2.5, 30: 1.5,
}
# Default weight for any other d/f metal
DEFAULT_DF_WEIGHT = 2.0


def metal_score(atomic_nums: List[int]) -> float:
    score = 0.0
    for z in atomic_nums:
        if z in TRANSITION_Z or z in LANTHANIDES_Z or z in ACTINIDES_Z:
            score += METAL_WEIGHTS.get(z, DEFAULT_DF_WEIGHT)
    return score


def feature_pack(mol, sdf_text: Optional[str]=None) -> Dict[str, float]:
    """RDKit'ten kestirim için özellikleri çıkar."""
    if mol is None or not RDKit_AVAILABLE:
        # RDKit yoksa SDF metninden temel öznitelikleri çıkar
        return parse_sdf_basic_features(sdf_text)
    atoms = list(mol.GetAtoms())
    atomic_nums = [a.GetAtomicNum() for a in atoms]
    heavy = sum(1 for z in atomic_nums if z > 1)
    hetero = sum(1 for a in atoms if a.GetAtomicNum() in (7, 8, 16))
    rings = mol.GetRingInfo().NumRings()
    charge = sum(abs(a.GetFormalCharge()) for a in atoms)
    arom_atoms = sum(1 for a in atoms if a.GetIsAromatic())
    arom_frac = (arom_atoms / max(1, heavy))
    # conjugated bonds fraction
    bonds = list(mol.GetBonds())
    conj_bonds = sum(1 for b in bonds if b.GetIsConjugated())
    conj = conj_bonds / max(1, len(bonds))
    # radicals (if encoded)
    try:
        radicals = float(mol.GetNumRadicalElectrons() or 0)
    except Exception:
        radicals = 0.0
    mscore = metal_score(atomic_nums)
    return {"radicals": radicals, "metal": mscore, "conj": conj, "arom_frac": arom_frac,
            "hetero": float(hetero), "rings": float(rings), "charge": float(charge), "heavy": float(heavy)}


def estimate_relaxivity(mol, props: Dict[str, str], sdf_text: Optional[str]=None) -> Tuple[Relaxivity, Dict[str, float]]:
    """SDF property'lerinde r1/r2 varsa doğrudan kullan; yoksa öznitelik skorundan tahmin."""
    # 1) SDF property override
    def _as_float(x):
        try:
            return float(str(x).strip())
        except Exception:
            return None
    r1_prop = _as_float((props or {}).get("r1"))
    r2_prop = _as_float((props or {}).get("r2"))
    if r1_prop is not None or r2_prop is not None:
        r1 = r1_prop if r1_prop is not None else 0.3
        r2 = r2_prop if r2_prop is not None else 0.4
        return Relaxivity(r1=max(0.0, r1), r2=max(0.0, r2)), {"source": 1}

    # 2) Heuristic from features
    f = feature_pack(mol, sdf_text)
    # para_score: ağırlıklı toplam
    para = 0.0
    para += 1.2 * f["radicals"]            # radikal e−
    para += 1.0 * f["metal"]               # metal ağırlığı
    para += 1.1 * f["conj"] * (1.0 + 0.5*f["arom_frac"]) * (1.0 + 0.02*f["heavy"])  # konjugasyon × aromatiklik × boyut
    para += 0.2 * f["hetero"]              # N/O/S donor/akseptörler
    para += 0.1 * f["rings"]               # halka sayısı
    para += 0.15 * f["charge"]             # formel yük

    # r1/r2 eşleme (saturasyon sınırı)
    r1 = 0.15 + 1.05 * para
    r2 = 0.20 + 1.20 * para
    # mantıklı sınırlar
    r1 = float(np.clip(r1, 0.0, 30.0))
    r2 = float(np.clip(r2, 0.0, 30.0))

    # Gd varsa r1/r2'yi yükselt (metal_score zaten arttırır; burada üst tavanı zorlarız)
    if RDKit_AVAILABLE and mol is not None:
        if any(a.GetAtomicNum() == 64 for a in mol.GetAtoms()):
            r1 = max(r1, GD_R1)
            r2 = max(r2, GD_R2)

    f_out = dict(f)
    f_out.update({"para_score": para, "r1_est": r1, "r2_est": r2, "source": 2})
    return Relaxivity(r1=r1, r2=r2), f_out

# ------------------------------
# MR sinyal modeli
# ------------------------------

def apply_relaxivity(T10_ms: float, T20_ms: float, r1: float, r2: float, conc_mM: float) -> Tuple[float, float]:
    T10 = max(T10_ms, 1e-6) / 1000.0
    T20 = max(T20_ms, 1e-6) / 1000.0
    R1 = 1.0 / T10 + r1 * conc_mM
    R2 = 1.0 / T20 + r2 * conc_mM
    T1_s = 1.0 / max(R1, 1e-9)
    T2_s = 1.0 / max(R2, 1e-9)
    return T1_s * 1000.0, T2_s * 1000.0


def spin_echo_signal(TR_ms: float, TE_ms: float, T1_ms: float, T2_ms: float, PD: float = 1.0) -> float:
    TR = max(TR_ms, 1e-9)
    TE = max(TE_ms, 1e-9)
    T1 = max(T1_ms, 1e-9)
    T2 = max(T2_ms, 1e-9)
    return float(PD * (1.0 - math.exp(-TR / T1)) * math.exp(-TE / T2))

# ------------------------------
# Görüntüleme yardımcıları
# ------------------------------

def make_circle_mask(h: int, w: int, center: Position, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:h, :w]
    cy, cx = center
    dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
    return dist2 <= radius ** 2


def render_phantom(signals: Dict[str, float], size: Tuple[int, int] = (900, 1200), noise_sigma: float = IMG_NOISE_SIGMA,
                   normalize: bool = NORMALIZE_IMAGE) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, Position], int]:
    h, w = size
    img = np.zeros((h, w), dtype=np.float32)
    rows, cols = 4, 2
    pad_y, pad_x = 40, 40
    cell_h = (h - (rows + 1) * pad_y) // rows
    cell_w = (w - (cols + 1) * pad_x) // cols
    radius = int(min(cell_h, cell_w) * 0.35)
    centers: Dict[str, Position] = {}
    masks: Dict[str, np.ndarray] = {}
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

    # Normalizasyonu daha duyarlı yap: 98. yüzdelik ile ölçekle
    if normalize:
        nonzero = img[img > 0]
        if nonzero.size > 0:
            scale = float(np.percentile(nonzero, 98))
            if scale > 1e-9:
                img = img / scale
    img = np.clip(img, 0.0, 1.0)

    # Gamma eşlemesi: düşük sinyalleri vurgula (DISPLAY_GAMMA<1 daha beyaz)
    try:
        if DISPLAY_GAMMA and DISPLAY_GAMMA > 0:
            img = (img ** DISPLAY_GAMMA).astype(np.float32)
    except Exception:
        pass

    # Gaussian gürültü (MR benzeri)
    if noise_sigma > 0:
        noise = np.random.normal(loc=0.0, scale=noise_sigma, size=img.shape).astype(np.float32)
        img = np.clip(img + noise, 0.0, 1.0)

    # Vinyet (hafif)
    Y, X = np.indices(img.shape)
    Y = (Y - h / 2) / (h / 2)
    X = (X - w / 2) / (w / 2)
    rad = np.sqrt(X**2 + Y**2)
    vignette = np.clip(1.0 - 0.1 * rad, 0.9, 1.0)
    img = np.clip(img * vignette, 0.0, 1.0)

    return img, masks, centers, radius

# ==============================
# UI (yalın: SDF + TR/TE)
# ==============================

st.title("MRI Phantom Simulator (SDF)")
st.caption("Sadece SDF + TR/TE ver; r1/r2 otomatik kestirilsin. (Sezgisel/öğretici in‑silico model)")

with st.sidebar:
    st.header("Molekül (SDF)")
    sdf_file = st.file_uploader("Sadece .sdf", type=["sdf"], accept_multiple_files=False)
    mol_info = None
    est_info: Dict[str, float] = {}
    test_relax = Relaxivity(r1=0.5, r2=0.5)  # fallback
    if sdf_file is not None:
        sdf_bytes = sdf_file.read()
        mol_info = parse_sdf(sdf_bytes)
        st.success("SDF yüklendi.")
        if mol_info.get("image") is not None:
            st.image(mol_info["image"], caption=mol_info.get("title") or "Molekül", use_column_width=True)
        else:
            st.write(mol_info.get("title") or "Molekül")
            if mol_info.get("formula"): st.write(f"Formül: {mol_info['formula']}")
            if mol_info.get("mw"): st.write(f"Mol kütlesi: {mol_info['mw']} g/mol")
        test_relax, est_info = estimate_relaxivity(mol_info.get("mol"), mol_info.get("props", {}), mol_info.get("sdf_text"))
        st.markdown(f"**r1/r2 (otomatik):** `{test_relax.r1:.2f}` / `{test_relax.r2:.2f}` s⁻¹·mM⁻¹")
        if est_info.get("source") == 1:
            st.caption("SDF property bloklarından alındı (r1/r2 sağlandı).")
        else:
            # küçük özet
            st.caption("Heuristik kestirim — öznitelik özeti:")
            show = {k: est_info[k] for k in ["para_score","radicals","metal","conj","arom_frac","hetero","rings","charge"] if k in est_info}
            st.json(show)
    else:
        st.info("Başlamak için bir SDF yükleyin.")

# Zamanlama (yalın)
st.sidebar.markdown("---")
st.sidebar.header("Zamanlama")
mode = st.sidebar.radio("Simülasyon modu", ["TR taraması (TE sabit)", "TE taraması (TR sabit)"], index=0)
if mode.startswith("TR"):
    TR_list_str = st.sidebar.text_input("TR listesi (ms)", value="75,150,300,600,1200,2400,4800")
    TR_list = [float(x) for x in TR_list_str.replace(";", ",").split(",") if x.strip()]
    if not TR_list:
        TR_list = [75,150,300,600,1200,2400,4800]
    TR_sel = st.sidebar.select_slider("Seçili TR (ms)", options=TR_list, value=TR_list[min(2, len(TR_list)-1)])
    TE_fixed = st.sidebar.number_input("TE (ms)", 1.0, 1000.0, 80.0, 1.0)
else:
    TE_list_str = st.sidebar.text_input("TE listesi (ms)", value="10,20,40,80,120")
    TE_list = [float(x) for x in TE_list_str.replace(";", ",").split(",") if x.strip()]
    if not TE_list:
        TE_list = [10,20,40,80,120]
    TE_sel = st.sidebar.select_slider("Seçili TE (ms)", options=TE_list, value=TE_list[min(2, len(TE_list)-1)])
    TR_fixed = st.sidebar.number_input("TR (ms)", 1.0, 10000.0, 600.0, 10.0)

# ==============================
# Ana akış
# ==============================
left, right = st.columns([0.6, 0.4])

with left:
    st.subheader("Fantom görüntüsü (2×4)")
    if sdf_file is None:
        st.warning("Simülasyon için bir **SDF** yükleyin.")

    base = Baseline(T10_ms=DEFAULT_T10_MS, T20_ms=DEFAULT_T20_MS, PD=DEFAULT_PD)
    gd_relax = Relaxivity(r1=GD_R1, r2=GD_R2)

    conc_map: Dict[str, float] = {
        "1":      STOCK_MOLAR_M * 1.0,
        "1/2":    STOCK_MOLAR_M * 0.5,
        "1/4":    STOCK_MOLAR_M * 0.25,
        "1/8":    STOCK_MOLAR_M * 0.125,
        "1/16":   STOCK_MOLAR_M * 0.0625,
        "1/32":   STOCK_MOLAR_M * 0.03125,
        "Gd":     GD_MOLAR_M,
        "Saf Su": 0.0,
    }

    def compute_signals_for(TR_ms: float, TE_ms: float) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        signals: Dict[str, float] = {}
        rows: List[Dict[str, float]] = []
        for label, _ in LAYOUT:
            if label == "Gd":
                T1_ms, T2_ms = apply_relaxivity(base.T10_ms, base.T20_ms, gd_relax.r1, gd_relax.r2, conc_map[label])
            elif label == "Saf Su":
                T1_ms, T2_ms = base.T10_ms, base.T20_ms
            else:
                T1_ms, T2_ms = apply_relaxivity(base.T10_ms, base.T20_ms, test_relax.r1, test_relax.r2, conc_map[label])
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
    img, masks, centers, radius = render_phantom(signals)

    fig, ax = plt.subplots(figsize=(8, 10))
    ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
    ax.axis("off")
    for label, (cy, cx) in centers.items():
        ax.text(cx, cy, label, ha="center", va="center", fontsize=12, color="w")
    st.pyplot(fig, use_container_width=True)

    # T1/T2 tablosu (PNG'nin altında)
    st.markdown("### T1/T2 (ms) Tablosu — Fantomlar")
    order = [lbl for lbl, _ in LAYOUT]
    df_relax = pd.DataFrame(rows)[["Phantom", "Kons. (mM)", "T1 (ms)", "T2 (ms)"]]
    df_relax["_ord"] = df_relax["Phantom"].apply(lambda x: order.index(x) if x in order else 999)
    df_relax = df_relax.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    st.dataframe(df_relax, use_container_width=True, hide_index=True)

    # --- Relaksivite Grafikleri ---
    st.markdown("### Relaksivite Grafikleri")
    try:
        plot_df = df_relax[~df_relax["Phantom"].isin(["Saf Su", "Gd"])].copy()
        C = plot_df["Kons. (mM)"].astype(float).values
        T1v = plot_df["T1 (ms)"].astype(float).values
        T2v = plot_df["T2 (ms)"].astype(float).values
        # (Kaldırıldı) T1/T2 vs Konsantrasyon grafiklerini göstermiyoruz; yalnızca relaksivite (ΔR1/ΔR2) fitleri aşağıda çiziliyor.
        

        # ΔR1/ΔR2 (s^-1) vs Konsantrasyon ve doğrusal fit (relaksivite)
        R1 = 1000.0 / np.maximum(T1v, 1e-9)
        R2 = 1000.0 / np.maximum(T2v, 1e-9)
        R10 = 1000.0 / max(base.T10_ms, 1e-9)
        R20 = 1000.0 / max(base.T20_ms, 1e-9)
        dR1 = R1 - R10
        dR2 = R2 - R20
        denom = max(float((C**2).sum()), 1e-12)
        r1_fit = float((C * dR1).sum() / denom)
        r2_fit = float((C * dR2).sum() / denom)

        xline = np.linspace(0.0, float(C.max()) * 1.05 if C.size else 1.0, 100)

        fig_r1, ax_r1 = plt.subplots(figsize=(6, 3.5))
        ax_r1.scatter(C, dR1)
        ax_r1.plot(xline, r1_fit * xline)
        ax_r1.set_xlabel("Konsantrasyon (mM)")
        ax_r1.set_ylabel("ΔR1 = 1/T1 − 1/T1₀ (s⁻¹)")
        ax_r1.set_title("Relaksivite r1 — doğrusal fit (kesişim 0)")
        ax_r1.grid(True, alpha=0.3)
        st.pyplot(fig_r1, use_container_width=True)
        st.caption(f"Tahmini r1 (eğim): **{r1_fit:.2f} s⁻¹·mM⁻¹**")

        fig_r2, ax_r2 = plt.subplots(figsize=(6, 3.5))
        ax_r2.scatter(C, dR2)
        ax_r2.plot(xline, r2_fit * xline)
        ax_r2.set_xlabel("Konsantrasyon (mM)")
        ax_r2.set_ylabel("ΔR2 = 1/T2 − 1/T2₀ (s⁻¹)")
        ax_r2.set_title("Relaksivite r2 — doğrusal fit (kesişim 0)")
        ax_r2.grid(True, alpha=0.3)
        st.pyplot(fig_r2, use_container_width=True)
        st.caption(f"Tahmini r2 (eğim): **{r2_fit:.2f} s⁻¹·mM⁻¹**")
    except Exception as e:
        st.warning(f"Relaksivite grafikleri çizilemedi: {e}")

    # PNG export
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=200)
    buf.seek(0)
    st.download_button("PNG indir", data=buf, file_name="phantom_sim.png", mime="image/png")

with right:
    st.subheader("ROI & Parametre Özeti")
    df = pd.DataFrame(rows)
    order = [lbl for lbl, _ in LAYOUT]
    df["_ord"] = df["Phantom"].apply(lambda x: order.index(x) if x in order else 999)
    df = df.sort_values("_ord").drop(columns=["_ord"]).reset_index(drop=True)
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("ROI CSV indir", data=csv, file_name=f"roi_TR{int(TR_curr)}_TE{int(TE_curr)}.csv", mime="text/csv")

st.markdown("""
---
**Yerleşim:**
```
1/4 | 1/8
1/2 | 1/16
1   | 1/32
Gd  | Saf Su
```
**Model:** S = PD · (1 − e^(−TR/T1)) · e^(−TE/T2)
**Not:** r1/r2, SDF’ten otomatik kestirilir (özellikler → para_score → r1/r2). SDF property içinde r1/r2 varsa doğrudan kullanılır.
""")
