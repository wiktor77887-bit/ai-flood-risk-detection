import json
import os

import folium
import matplotlib.pyplot as plt
import numpy as np
import requests
import segmentation_models_pytorch as smp
import streamlit as st
import torch
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Analiza powodziowa AI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from ui_common import wymagaj_logowania, nawigacja
wymagaj_logowania()
nawigacja()

# ==========================================
# PALETA
# ==========================================
INK = "#12303F"        # tekst, granice
MUTED = "#4A6572"      # tekst drugorzędny
WATER = "#1F7A8C"      # akcent wody
MIST = "#EAF2F5"       # tło kart
PAPER = "#F6F9FA"      # tło strony
ALERT = "#E4572E"      # markery / wykryte obiekty

# ==========================================
# STYLE (jawne kolory - działają też w trybie ciemnym przeglądarki)
# ==========================================
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');

    html, body, [class*="css"], .stApp {{
        font-family: 'Manrope', sans-serif;
    }}
    .stApp {{ background: {PAPER}; }}
    header[data-testid="stHeader"] {{ background: transparent; }}
    .block-container {{ padding-top: 2rem; max-width: 1280px; }}

    /* Tekst */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ color: {INK}; }}
    .stApp p, .stApp label, .stApp li {{ color: {INK}; }}
    [data-testid="stCaptionContainer"] p {{ color: {MUTED}; }}

    .hero h1 {{
        font-size: 2.6rem; font-weight: 800; letter-spacing: -0.02em;
        margin: 0 0 .35rem 0;
    }}
    .hero p {{
        font-size: 1.05rem; color: {MUTED}; margin: 0 0 1.2rem 0; max-width: 46rem;
    }}

    .city-card {{
        background: transparent; padding: .1rem .2rem .5rem .2rem; margin: 0;
    }}
    .city-card h3 {{ margin: 0 0 .25rem 0; font-size: 1.15rem; }}
    .city-card p  {{ margin: 0; color: {MUTED}; font-size: .93rem; line-height: 1.45; }}

    .badge {{
        display: inline-block; padding: .15rem .6rem; border-radius: 999px;
        font-size: .78rem; font-weight: 700; background: {ALERT}; color: white;
        margin-left: .5rem; vertical-align: middle;
    }}

    /* Metryki */
    div[data-testid="stMetric"] {{
        background: white; border: 1px solid #D5E3E9; border-radius: 12px;
        padding: .9rem 1.1rem;
    }}
    div[data-testid="stMetricLabel"], div[data-testid="stMetricLabel"] p {{
        color: {MUTED} !important; font-weight: 500;
    }}
    div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] div {{
        font-weight: 800; color: {INK} !important;
    }}

    /* Zakładki */
    .stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
    .stTabs button[data-baseweb="tab"] {{
        height: 2.8rem; padding: 0 1.2rem; border-radius: 10px 10px 0 0;
    }}
    .stTabs button[data-baseweb="tab"] p {{
        color: {MUTED} !important; font-weight: 700;
    }}
    .stTabs button[data-baseweb="tab"][aria-selected="true"] p {{
        color: {WATER} !important;
    }}
    .stTabs [data-baseweb="tab-highlight"] {{ background-color: {WATER} !important; }}

    /* Przyciski (nowe i stare wersje Streamlit) */
    .stButton > button,
    [data-testid^="stBaseButton"] {{
        border-radius: 10px; border: 1px solid {WATER} !important;
        background: white !important; color: {WATER} !important;
    }}
    .stButton > button p,
    [data-testid^="stBaseButton"] p {{
        color: {WATER} !important; font-weight: 700;
    }}
    .stButton > button:hover,
    [data-testid^="stBaseButton"]:hover {{
        background: {WATER} !important;
    }}
    .stButton > button:hover p,
    [data-testid^="stBaseButton"]:hover p {{
        color: white !important;
    }}

    iframe {{ border-radius: 16px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 1. MODEL AI
# ==========================================
@st.cache_resource
def load_unet_model():
    device = torch.device("cpu")
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    if os.path.exists("best_model.pth"):
        ckpt = torch.load("best_model.pth", map_location=device, weights_only=False)
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict)
    model.eval()
    return model, device


# Cache zależy od czasu modyfikacji pliku: po ponownym uruchomieniu precompute.py
# aplikacja nie używa starych wyników bez restartu.
def mtime(plik: str) -> float:
    return os.path.getmtime(plik)


@st.cache_data(show_spinner=False)
def _liczba_patchy(plik_npy: str, mt: float) -> int:
    a = np.load(plik_npy, mmap_mode="r")
    return 1 if a.ndim == 2 else a.shape[0]   # 2D = pojedynczy raster


def liczba_patchy(plik_npy: str) -> int:
    return _liczba_patchy(plik_npy, mtime(plik_npy))


def run_segmentation(plik_npy: str, idx: int):
    return _run_segmentation(plik_npy, idx, mtime(plik_npy))


@st.cache_data(show_spinner=False)
def _run_segmentation(plik_npy: str, idx: int, mt: float):
    """Zwraca (dane surowe, maska) dla jednego wycinka 256x256."""
    # mmap: z pliku 500-900 MB czytamy tylko jeden wycinek
    arr = np.load(plik_npy, mmap_mode="r")
    raw = arr if arr.ndim == 2 else arr[idx]
    data = np.squeeze(np.array(raw, dtype=np.float32))
    if data.ndim != 2:
        raise ValueError(f"Nieoczekiwany kształt danych w {plik_npy}: {arr.shape}")

    if data.max() > 1.0 or data.min() < 0.0:
        p2, p98 = np.percentile(data, (2, 98))
        norm = np.clip((data - p2) / (p98 - p2 + 1e-6), 0, 1)
    else:
        norm = data.copy()

    model, device = load_unet_model()
    H, W = norm.shape
    pad_h = (32 - (H % 32)) % 32
    pad_w = (32 - (W % 32)) % 32

    t = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect").to(device)

    with torch.no_grad():
        prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()

    mask = (prob[:H, :W] > 0.5).astype(np.uint8)
    return data, mask


@st.cache_data(show_spinner=False)
def _uklad_siatki(sciezka: str, mt: float):
    """Z pliku pozycji: {(kolumna, wiersz): numer wycinka} i odwrotnie {numer: (kolumna, wiersz)}.
    Kolumny rosną na wschód, wiersze na południe. Które pole "siatka" jest kolumną, a które
    wierszem, wykrywamy z położenia narożników ("rogi"), więc nie zależy to od kolejności zapisu."""
    poz = _wczytaj_json(sciezka, mt)
    # pozycje_patchy.py zapisuje numer (wiersz, kolumna) każdego wycinka w "rzad_kolumna";
    # "siatka" to tam tylko rozmiar siatki [wiersze, kolumny] (starsze pliki: lista par w "siatka").
    siatka, rogi = poz.get("rzad_kolumna"), poz.get("rogi")
    if not siatka:
        siatka = poz.get("siatka")
    if (not siatka or not rogi or len(siatka) != len(rogi)
            or not isinstance(siatka[0], (list, tuple))):
        return None
    pos = {tuple(s): i for i, s in enumerate(siatka) if rogi[i]}
    kroki = {0: [], 1: []}
    for (a, b), i in pos.items():
        for ax, sas in ((0, (a + 1, b)), (1, (a, b + 1))):
            j = pos.get(sas)
            if j is not None:
                kroki[ax].append((rogi[j][0][0] - rogi[i][0][0], rogi[j][0][1] - rogi[i][0][1]))
    if not kroki[0] or not kroki[1]:
        return None
    m = {ax: np.median(np.array(v), axis=0) for ax, v in kroki.items()}
    kol_ax, wier_ax = (0, 1) if abs(m[0][0]) >= abs(m[1][0]) else (1, 0)
    sx = 1 if m[kol_ax][0] >= 0 else -1      # kolumny: rosnąca długość geogr. = wschód
    sy = 1 if m[wier_ax][1] < 0 else -1      # wiersze: malejąca szerokość geogr. = południe
    uklad, odwr = {}, {}
    for v, i in pos.items():
        kw = (sx * v[kol_ax], sy * v[wier_ax])
        uklad[kw] = i
        odwr[i] = kw
    return uklad, odwr


def indeksy_bloku(pozycje_plik: str, idx: int, k: int):
    """Numery wycinków bloku k x k, w którym leży wycinek idx (wiersz po wierszu, od północnego
    zachodu). Bloki tworzą stałą siatkę (nie nakładają się): wiersz i kolumna dzielone przez k.
    None na liście = brak wycinka w tym miejscu."""
    u = _uklad_siatki(pozycje_plik, os.path.getmtime(pozycje_plik))
    if not u or idx not in u[1]:
        return None
    uklad, odwr = u
    c, r = odwr[idx]
    c0, r0 = (c // k) * k, (r // k) * k
    return tuple(uklad.get((c0 + dx, r0 + dy)) for dy in range(k) for dx in range(k))


OPCJE_BLOKU = {"Mały (2 × 2)": 2, "Duży (4 × 4)": 4}


def wybor_bloku(miasto: str, klucz: str, opis: str) -> int:
    """Radio z rozmiarem bloku. Wartość pamiętana osobno (session_state), więc ta sama w widokach
    „Położenie” i „Detekcja budynków” (Streamlit kasuje stan widgetu, którego nie narysowano)."""
    skl = f"k_{miasto}"
    wartosci = list(OPCJE_BLOKU.values())
    biezace = st.session_state.get(skl, 2)
    if biezace not in wartosci:
        biezace = 2
    wyb = st.radio(opis, list(OPCJE_BLOKU), index=wartosci.index(biezace),
                   horizontal=True, key=f"{klucz}_{miasto}")
    st.session_state[skl] = OPCJE_BLOKU[wyb]
    return OPCJE_BLOKU[wyb]


def sciezki_rastra(plik_npy: str):
    """(raster .npy, meta .json) zapisane przez raster_pelny.py obok patches_<miasto>_nmt.npy."""
    baza = os.path.splitext(plik_npy)[0].rsplit("_", 1)[0]
    return baza + "_raster.npy", baza + "_raster.json"


def raster_pelny_ok(plik_npy: str, pozycje) -> bool:
    """Czy jest pełny raster (bez luk) zgodny z plikiem pozycji (ta sama siatka kafelków)."""
    npy_r, json_r = sciezki_rastra(plik_npy)
    if not (os.path.exists(npy_r) and os.path.exists(json_r) and pozycje):
        return False
    meta = _wczytaj_json(json_r, os.path.getmtime(json_r))
    return list(meta.get("siatka", [])) == list(pozycje.get("siatka", [None]))


def _segmentuj(norm: np.ndarray) -> np.ndarray:
    """U-Net na znormalizowanym obrazie 0..1 (z paddingiem do wielokrotności 32)."""
    model, device = load_unet_model()
    H, W = norm.shape
    pad_h = (32 - (H % 32)) % 32
    pad_w = (32 - (W % 32)) % 32
    t = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect").to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    return (prob[:H, :W] > 0.5).astype(np.uint8)


def run_segmentation_raster(plik_npy: str, idx: int, k: int, pozycje_plik: str):
    """Blok k x k (stała siatka) z pełnego rastra, bez luk; blok zawiera wycinek idx."""
    u = _uklad_siatki(pozycje_plik, os.path.getmtime(pozycje_plik))
    c, r = u[1][idx]
    npy_r, _ = sciezki_rastra(plik_npy)
    return _run_segmentation_raster(npy_r, (r // k) * k, (c // k) * k, k, mtime(npy_r))


@st.cache_data(show_spinner=False, max_entries=6)
def _run_segmentation_raster(plik_raster: str, r0: int, c0: int, k: int, mt: float):
    arr = np.load(plik_raster, mmap_mode="r")        # (H, W) uint16, wartość / 65535 = norma 0..1
    P = 256
    y0, x0 = r0 * P, c0 * P
    y1, x1 = min(y0 + k * P, arr.shape[0]), min(x0 + k * P, arr.shape[1])
    data = np.full((k * P, k * P), np.nan, dtype=np.float32)   # NaN tylko za brzegiem rastra
    data[:y1 - y0, :x1 - x0] = np.asarray(arr[y0:y1, x0:x1], dtype=np.float32) / 65535.0
    brak = np.isnan(data)
    norm = np.where(brak, 0.0, data).astype(np.float32)
    mask = _segmentuj(norm)
    mask[brak] = 0
    return data, mask


def run_segmentation_blok(plik_npy: str, indeksy: tuple, k: int):
    return _run_segmentation_blok(plik_npy, indeksy, k, mtime(plik_npy))


@st.cache_data(show_spinner=False)
def _run_segmentation_blok(plik_npy: str, indeksy: tuple, k: int, mt: float):
    """Skleja k x k sąsiednich wycinków w jeden obraz i robi na nim segmentację.
    Zwraca (dane z NaN w miejscach bez wycinka, maska)."""
    arr = np.load(plik_npy, mmap_mode="r")
    P = arr.shape[-1]
    data = np.full((k * P, k * P), np.nan, dtype=np.float32)
    for n_, j in enumerate(indeksy):
        if j is None:
            continue
        dy, dx = divmod(n_, k)
        data[dy * P:(dy + 1) * P, dx * P:(dx + 1) * P] = np.squeeze(np.array(arr[j], dtype=np.float32))
    brak = np.isnan(data)
    wart = data[~brak]
    if wart.size == 0:
        raise ValueError("Wszystkie wycinki bloku są puste.")
    if wart.max() > 1.0 or wart.min() < 0.0:
        p2, p98 = np.percentile(wart, (2, 98))        # wspólna normalizacja całego bloku (bez szwów)
        norm = np.clip((data - p2) / (p98 - p2 + 1e-6), 0, 1)
    else:
        norm = data.copy()
    norm[brak] = 0.0

    model, device = load_unet_model()
    H, W = norm.shape
    pad_h = (32 - (H % 32)) % 32
    pad_w = (32 - (W % 32)) % 32
    t = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t = F.pad(t, (0, pad_w, 0, pad_h), mode="reflect").to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(t)).squeeze().cpu().numpy()
    mask = (prob[:H, :W] > 0.5).astype(np.uint8)
    mask[brak] = 0
    return data, mask


@st.cache_data(show_spinner=False)
def load_poland_geojson():
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/POL.geo.json"
    try:
        return requests.get(url, timeout=5).json()
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_poland_outline():
    """Dokladny obrys Polski (GUGiK PRG) z dane/obrysy.geojson; gdy go brak - prostszy z internetu."""
    sciezka = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dane", "obrysy.geojson")
    try:
        with open(sciezka, encoding="utf-8") as f:
            for feat in json.load(f)["features"]:
                if feat["properties"].get("typ") == "polska":
                    return feat
    except Exception:
        pass
    return load_poland_geojson()


def _pierscienie(geo):
    """Zewnetrzne pierscienie obrysu jako listy [lat, lon] (do maski wokol Polski)."""
    g = geo.get("geometry", geo) if isinstance(geo, dict) else geo
    if g.get("type") == "FeatureCollection":
        g = g["features"][0]["geometry"]
    polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
    return [[[c[1], c[0]] for c in p[0]] for p in polys]


# ==========================================
# 2. AGLOMERACJE
# ==========================================
def first_existing(paths):
    """Pierwszy istniejacy plik z listy (nazwy bywaja nmt/nmpt)."""
    if isinstance(paths, str):
        paths = [paths]
    for p in paths:
        if os.path.exists(p):
            return p
    return None


# npy  - lista mozliwych nazw patchy NMT/NMPT (moze byc pusta => tylko symulacja)
# gify - {etykieta: plik}; kazde miasto moze miec kilka symulacji
MIASTA = {
    "Poznań": {
        "lat": 52.4064, "lon": 16.9252,
        "npy": ["patches_poznan_nmt.npy", "patches_poznan_nmpt.npy"],
        "gify": {"Warta": "powodz_warta_poznan.gif"},
        "patch": 0,   # <- numer wycinka użytego w Animacja.py
        "opis": "Dolina Warty, zróżnicowana gęstość zabudowy miejskiej.",
    },
    "Warszawa": {
        "lat": 52.2297, "lon": 21.0122,
        "npy": ["patches_warszawa_nmt.npy", "patches_warszawa_nmpt.npy"],
        "gify": {"Wisła": "powodz_wisla_warszawa.gif"},
        "patch": 0,   # <- numer wycinka użytego w animacji Warszawy
        "opis": "Szerokie koryto Wisły, infrastruktura krytyczna i wały przeciwpowodziowe.",
    },
    "Wrocław": {
        "lat": 51.1079, "lon": 17.0385,
        "npy": ["patches_wroclaw_nmt.npy", "patches_wroclaw_nmpt.npy"],
        "gify": {"Odra": "powodz_odra_wroclaw.gif"},
        "patch": 0,
        "opis": "Wrocławski Węzeł Wodny na Odrze, historyczne centrum w zasięgu wielkich wezbrań.",
    },
    "Bydgoszcz": {
        "lat": 53.1235, "lon": 18.0084,
        "npy": ["patches_bydgoszcz_nmt.npy", "patches_bydgoszcz_nmpt.npy"],
        "gify": {"Wisła": "powodz_brda_bydgoszcz.gif"},
        "patch": 0,
        "opis": "Dolina Brdy i Wisły, Wyspa Młyńska i zabytkowe śródmieście nad wodą.",
    },
}

if "wybrane_miasto" not in st.session_state:
    st.session_state.wybrane_miasto = None
if "map_key" not in st.session_state:
    st.session_state.map_key = 0


def wybierz(miasto):
    st.session_state.wybrane_miasto = miasto
    st.rerun()


def wroc():
    st.session_state.wybrane_miasto = None
    st.session_state.map_key += 1  # świeża mapa, bez starego kliknięcia
    st.rerun()


# ==========================================
# 3. MAPA
# ==========================================
PULSE_CSS = f"""
<style>
.pin {{
    width: 18px; height: 18px; border-radius: 50%;
    background: {ALERT}; border: 3px solid white;
    box-shadow: 0 0 0 2px {ALERT}, 0 2px 6px rgba(0,0,0,.35);
    position: relative;
}}
.pin::after {{
    content: ""; position: absolute; inset: -9px; border-radius: 50%;
    border: 2px solid {ALERT}; opacity: 0;
    animation: pulse 2.2s ease-out infinite;
}}
@keyframes pulse {{
    0%   {{ transform: scale(.5); opacity: .8; }}
    100% {{ transform: scale(1.6); opacity: 0; }}
}}
@media (prefers-reduced-motion: reduce) {{ .pin::after {{ animation: none; }} }}
.leaflet-container {{ background: {PAPER}; }}
.leaflet-tooltip.city-tip {{
    font-family: 'Manrope', sans-serif; font-weight: 800; font-size: 14px;
    color: {INK}; border: none; border-radius: 8px;
    padding: 4px 10px; box-shadow: 0 2px 8px rgba(0,0,0,.22);
}}
.leaflet-tooltip-right.city-tip::before {{ border-right-color: white; }}
.leaflet-tooltip-left.city-tip::before {{ border-left-color: white; }}
</style>
"""


def zbuduj_mape_polski():
    m = folium.Map(
        location=[52.0, 19.3],
        zoom_start=6,
        tiles=None,
        zoom_control=False,
        zoom_snap=0.1,
        scrollWheelZoom=False,
        dragging=False,
        doubleClickZoom=False,
        touchZoom=False,
        boxZoom=False,
        keyboard=False,
        attributionControl=False,
    )
    m.get_root().html.add_child(folium.Element(PULSE_CSS))

    geo = load_poland_outline()
    if geo:
        nieklik = {"interactive": False}
        # przygaszenie wszystkiego poza Polską: duży prostokąt z "dziurą" w kształcie kraju
        swiat = [[-85, -180], [-85, 180], [85, 180], [85, -180]]
        try:
            folium.Polygon(locations=[swiat] + _pierscienie(geo), color=PAPER, weight=0, fill=True,
                           fill_color=PAPER, fill_opacity=0.82, **nieklik).add_to(m)
        except Exception:
            pass
        # granica: ciemna obwódka + biała linia, jak na stronie "Symulacja dla miejsca"
        for kolor, grub in ((INK, 5), ("#FFFFFF", 2.5)):
            folium.GeoJson(geo, style_function=lambda _, k=kolor, g=grub: {
                "color": k, "weight": g, "fill": False}, **nieklik).add_to(m)

    for nazwa, d in MIASTA.items():
        folium.Marker(
            location=[d["lat"], d["lon"]],
            tooltip=folium.Tooltip(nazwa, direction="left" if nazwa == "Warszawa" else "right",
                                   offset=(-12 if nazwa == "Warszawa" else 12, 0),
                                   permanent=True, class_name="city-tip"),
            icon=folium.DivIcon(
                html='<div class="pin"></div>',
                icon_size=(18, 18), icon_anchor=(9, 9),
            ),
        ).add_to(m)

    m.fit_bounds([[49.0, 14.1], [54.85, 24.15]], padding=(10, 10))
    return m


def mapa_miasta(nazwa):
    d = MIASTA[nazwa]
    m = folium.Map(
        location=[d["lat"], d["lon"]], zoom_start=11,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        attributionControl=False, zoom_control=True,
    )
    m.get_root().html.add_child(folium.Element(PULSE_CSS))
    folium.Marker(
        [d["lat"], d["lon"]],
        icon=folium.DivIcon(html='<div class="pin"></div>',
                            icon_size=(18, 18), icon_anchor=(9, 9)),
    ).add_to(m)
    return m


def sciezka_pozycji(plik_npy: str) -> str:
    return os.path.splitext(plik_npy)[0].rsplit("_", 1)[0] + "_pozycje.json"


@st.cache_data(show_spinner=False)
def _wczytaj_json(sciezka: str, mtime: float):
    with open(sciezka, encoding="utf-8") as f:
        return json.load(f)


def wczytaj_pozycje(plik_npy: str):
    """Pozycje wycinków zapisane przez pozycje_patchy.py (None, jeśli brak pliku).
    Numer wycinka w .npy to ranking zabudowy (precompute.py sortuje), nie pozycja.
    Klucz cache = czas modyfikacji pliku, więc nowy plik jest widoczny bez restartu."""
    sciezka = sciezka_pozycji(plik_npy)
    if not os.path.exists(sciezka):
        return None
    return _wczytaj_json(sciezka, os.path.getmtime(sciezka))


def _bloki_geojson(pozycje, odwr, k, pelne=False):
    """Obrysy bloków k x k (stała siatka) w prawdziwym położeniu. Narożniki liczone z kroku siatki
    jednego wycinka, więc działają też tam, gdzie część wycinków bloku nie istnieje."""
    rogi = pozycje["rogi"]
    ref = {}
    for i, (c, r) in odwr.items():
        if not rogi[i]:
            continue
        bk = (c // k, r // k)
        odl = (c - bk[0] * k) + (r - bk[1] * k)
        if bk not in ref or odl < ref[bk][0]:
            ref[bk] = (odl, i)
    ile = {}
    for i, (c, r) in odwr.items():
        if rogi[i]:
            ile[(c // k, r // k)] = ile.get((c // k, r // k), 0) + 1
    features = []
    for bk, (_, i) in ref.items():
        c, r = odwr[i]
        nw, ne, _se, sw = (np.array(x) for x in rogi[i][:4])
        dcol, drow = ne - nw, sw - nw
        nw_b = nw - (c - bk[0] * k) * dcol - (r - bk[1] * k) * drow
        pts = [nw_b, nw_b + k * dcol, nw_b + k * dcol + k * drow, nw_b + k * drow]
        ring = [[float(x), float(y)] for x, y in pts]
        features.append({
            "type": "Feature",
            "properties": {"idx": i, "bk": list(bk), "n": ile[bk], "pelny": pelne or ile[bk] == k * k,
                           "opis": (f"{k * 256 * pozycje.get('res_m', 0.5):.0f} × "
                                    f"{k * 256 * pozycje.get('res_m', 0.5):.0f} m" if pelne else
                                    f"{ile[bk]} z {k * k} wycinków" +
                                    ("" if ile[bk] == k * k else " (są puste pola)"))},
            "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]},
        })
    return features


def mapa_siatki(pozycje, wybrany, plik_poz=None, k=1, pelne=False):
    """Satelitarna mapa z kwadratami = wycinki 256x256 w ich prawdziwym położeniu.
    Dla k > 1 kwadraty są złączone w bloki k x k (jeden klik = jeden blok)."""
    features, lons, lats = [], [], []
    for i, rogi in enumerate(pozycje["rogi"]):
        if rogi:
            lons += [p[0] for p in rogi]
            lats += [p[1] for p in rogi]

    wyb_bk = None
    pola, aliasy = ["idx"], ["Wycinek:"]
    u = _uklad_siatki(plik_poz, os.path.getmtime(plik_poz)) if (plik_poz and k > 1) else None
    if u:
        features = _bloki_geojson(pozycje, u[1], k, pelne)
        if wybrany in u[1]:
            c, r = u[1][wybrany]
            wyb_bk = [c // k, r // k]
        pola, aliasy = ["opis"], ["Blok:"]
    else:
        for i, rogi in enumerate(pozycje["rogi"]):
            if rogi:
                features.append({
                    "type": "Feature",
                    "properties": {"idx": i},
                    "geometry": {"type": "Polygon", "coordinates": [rogi + [rogi[0]]]},
                })

    m = folium.Map(
        location=[(min(lats) + max(lats)) / 2, (min(lons) + max(lons)) / 2],
        zoom_start=14,
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri, Maxar, Earthstar Geographics",
        attributionControl=False, zoom_control=True,
    )
    m.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]])

    def styl(f):
        pr = f["properties"]
        wyb = (pr.get("bk") == wyb_bk) if "bk" in pr else (pr["idx"] == wybrany)
        niepelny = pr.get("pelny") is False
        return {
            "color": ALERT if wyb else "white",
            "weight": 3 if wyb else 1,
            "dashArray": "6 6" if (niepelny and not wyb) else None,
            "fillColor": ALERT if wyb else WATER,
            "fillOpacity": 0.45 if wyb else 0.08,
        }

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=styl,
        highlight_function=lambda f: {"weight": 3, "fillColor": WATER, "fillOpacity": 0.35},
        tooltip=folium.GeoJsonTooltip(fields=pola, aliases=aliasy),
    ).add_to(m)
    return m


# ==========================================
# 4. WIDOK 1 - MAPA POLSKI
# ==========================================
if st.session_state.wybrane_miasto is None:
    st.markdown(
        """
        <div class="hero">
            <h1>Zagrożenie powodziowe w polskich miastach</h1>
            <p>Model U-Net rozpoznaje budynki na numerycznym modelu terenu,
            a symulacja pokazuje, które z nich znajdą się pod wodą.
            Wybierz miasto na mapie albo z listy.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_map, col_list = st.columns([3, 1.3], gap="large")

    with col_map:
        map_data = st_folium(
            zbuduj_mape_polski(),
            height=600,
            use_container_width=True,
            key=f"mapa_{st.session_state.map_key}",
            returned_objects=["last_object_clicked_tooltip"],
        )
        tip = (map_data or {}).get("last_object_clicked_tooltip")
        if tip:
            for nazwa in MIASTA:
                if nazwa in tip:
                    wybierz(nazwa)

    with col_list:
        st.markdown("#### Miasta")
        for nazwa, d in MIASTA.items():
            badge = '<span class="badge">dane treningowe</span>' if d.get("trening") else ""
            with st.container(border=True):
                st.markdown(
                    f'<div class="city-card"><h3>{nazwa}{badge}</h3><p>{d["opis"]}</p></div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"Analizuj: {nazwa}", key=f"btn_{nazwa}", use_container_width=True,
                             type="primary"):
                    wybierz(nazwa)

# ==========================================
# 5. WIDOK 2 - DASHBOARD MIASTA
# ==========================================
else:
    wybrane = st.session_state.wybrane_miasto
    d = MIASTA[wybrane]

    top_l, top_r = st.columns([1, 5])
    with top_l:
        if st.button("← Mapa Polski", use_container_width=True):
            wroc()
    with top_r:
        badge = '<span class="badge">dane treningowe</span>' if d.get("trening") else ""
        st.markdown(
            f'<div class="hero"><h1 style="font-size:2rem;margin:0">{wybrane}{badge}</h1>'
            f'<p style="margin:.2rem 0 0 0">{d["opis"]}</p></div>',
            unsafe_allow_html=True,
        )

    if d.get("trening"):
        st.info(
            "Ten obszar był użyty do trenowania modelu, więc wynik segmentacji "
            "jest zawyżony. Do oceny jakości użyj pozostałych miast."
        )

    st.write("")

    npy_path = first_existing(d.get("npy", []))
    gify = {k: p for k, p in d.get("gify", {}).items() if os.path.exists(p)}
    n = liczba_patchy(npy_path) if npy_path else 0

    patch_key = f"patch_{wybrane}"
    widok_key = f"widok_{wybrane}"
    ver_key = f"ver_{wybrane}"
    if n == 1:
        st.session_state[patch_key] = 0   # pojedynczy raster - nie ma z czego wybierać
    idx = st.session_state.get(patch_key)

    if not npy_path and not gify:
        oczekiwane = list(d.get("npy", [])[:1]) + list(d.get("gify", {}).values())
        st.warning(f"Brak danych dla miasta **{wybrane}** na serwerze.")
        st.info(
            "Oczekiwane pliki: " + ", ".join(f"`{p}`" for p in oczekiwane)
            + ". Skopiuj je do folderu z aplikacją."
        )
    else:
        if not npy_path:
            st.info(
                "Dla tego miasta nie ma jeszcze danych NMT do detekcji budynków. "
                "Dostępna jest symulacja powodzi."
            )

        widoki = ["Położenie"] \
            + (["Detekcja budynków"] if npy_path else []) \
            + (["Symulacja powodzi"] if gify else [])

        # przełączenie widoku po kliknięciu kwadratu (przed utworzeniem radia)
        if st.session_state.get("_nowy_widok"):
            st.session_state[widok_key] = st.session_state.pop("_nowy_widok")

        # Zamiast st.tabs: Streamlit renderuje wszystkie zakładki naraz, więc GIF
        # startował od razu. Radio renderuje tylko wybrany widok.
        widok = st.radio("Widok", widoki, horizontal=True,
                         key=widok_key, label_visibility="collapsed")
        st.write("")

        # ---------- POŁOŻENIE (wybór kwadratu) ----------
        if widok == "Położenie":
            pozycje = wczytaj_pozycje(npy_path) if (npy_path and n > 1) else None
            if pozycje and pozycje.get("n") == n:
                plik_poz = sciezka_pozycji(npy_path)
                blok_ok = bool(_uklad_siatki(plik_poz, os.path.getmtime(plik_poz)))
                k = wybor_bloku(wybrane, "blok_poz", "Rozmiar obszaru (liczba wycinków)") if blok_ok else 1
                bok_m = k * pozycje.get("patch_px", 256) * pozycje.get("res_m", 0.5)
                ile_w = sum(1 for r in pozycje["rogi"] if r)
                pelne = bool(blok_ok and raster_pelny_ok(npy_path, pozycje))
                if k > 1 and pelne:
                    st.caption(f"Bloki {k} × {k} wycinków (ok. {bok_m:.0f} × {bok_m:.0f} m), bez luk. "
                               "Kliknij blok, aby przejść do detekcji budynków.")
                elif k > 1:
                    st.caption(f"{ile_w} wycinków 256×256 px złączonych w bloki {k} × {k} "
                               f"(ok. {bok_m:.0f} × {bok_m:.0f} m każdy). Kliknij blok, aby przejść do detekcji budynków.")
                else:
                    st.caption(f"Kliknij kwadrat na mapie ({ile_w} wycinków 256×256 px, "
                               f"ok. {bok_m:.0f} × {bok_m:.0f} m), aby przejść do detekcji budynków.")
                st.session_state.setdefault(ver_key, 0)
                map_data = st_folium(
                    mapa_siatki(pozycje, idx, plik_poz if blok_ok else None, k, pelne),
                    height=560,
                    use_container_width=True,
                    key=f"siatka_{wybrane}_{st.session_state[ver_key]}_{k}",
                    returned_objects=["last_active_drawing"],
                )
                klik = (map_data or {}).get("last_active_drawing")
                if klik:
                    wybrany = (klik.get("properties") or {}).get("idx")
                    if wybrany is not None:
                        st.session_state[patch_key] = int(wybrany)
                        st.session_state[ver_key] += 1   # świeża mapa, bez starego kliknięcia
                        st.session_state["_nowy_widok"] = "Detekcja budynków"
                        st.rerun()
            else:
                st_folium(
                    mapa_miasta(wybrane),
                    height=520,
                    use_container_width=True,
                    key=f"mini_{wybrane}",
                    returned_objects=[],
                )
                if npy_path and n > 1:
                    plik_poz = os.path.basename(sciezka_pozycji(npy_path))
                    if pozycje:
                        st.warning(
                            f"`{plik_poz}` opisuje {pozycje.get('n')} wycinków, a "
                            f"`{os.path.basename(npy_path)}` ma ich {n}. Pliki pochodzą z różnych "
                            "przebiegów precompute.py. Uruchom ponownie `python pozycje_patchy.py`. "
                            "Do tego czasu wpisz numer wycinka ręcznie."
                        )
                    else:
                        st.info(
                            "Brak pliku z pozycjami wycinków, więc nie da się pokazać siatki. "
                            f"Uruchom `python pozycje_patchy.py` i skopiuj `{plik_poz}` do folderu "
                            "z aplikacją. Do tego czasu wpisz numer wycinka ręcznie."
                        )
                    num = st.number_input("Numer wycinka", 0, n - 1, int(idx or 0),
                                          key=f"num_{wybrane}")
                    if st.button("Pokaż detekcję"):
                        st.session_state[patch_key] = int(num)
                        st.session_state["_nowy_widok"] = "Detekcja budynków"
                        st.rerun()
            st.caption("Mapa: Esri, Maxar, Earthstar Geographics")

        # ---------- DETEKCJA BUDYNKÓW ----------
        elif widok == "Detekcja budynków":
            if idx is None:
                st.info("Wybierz kwadrat w widoku „Położenie”.")
            else:
                poz = wczytaj_pozycje(npy_path) if n > 1 else None
                plik_poz = sciezka_pozycji(npy_path)
                blok_ok = bool(poz and poz.get("n") == n and os.path.exists(plik_poz)
                               and _uklad_siatki(plik_poz, os.path.getmtime(plik_poz)))
                k = 1
                if blok_ok:
                    k = wybor_bloku(wybrane, "blok_det", "Rozmiar obszaru (liczba wycinków)")
                if n > 1:
                    bok_m = k * poz.get("patch_px", 256) * poz.get("res_m", 0.5) if poz else None
                    opis_m = f", obszar ok. {bok_m:.0f} × {bok_m:.0f} m" if bok_m else ""
                    st.caption(f"Wybrany wycinek: {idx} z {n - 1} (256×256 px){opis_m}"
                               if k == 1 else
                               f"Blok {k} × {k} wycinków (zawiera wycinek {idx}){opis_m}")
                else:
                    st.caption("Plik zawiera jeden raster terenu (bez podziału na wycinki).")
                with st.spinner("Model przetwarza dane NMT…"):
                    if k == 1:
                        data, mask = run_segmentation(npy_path, idx)
                    elif raster_pelny_ok(npy_path, poz):
                        data, mask = run_segmentation_raster(npy_path, idx, k, plik_poz)
                        brakuje = 0
                    else:
                        ind = indeksy_bloku(plik_poz, idx, k)
                        data, mask = run_segmentation_blok(npy_path, ind, k)
                        brakuje = sum(1 for j in ind if j is None)
                        if brakuje:
                            st.caption(f"W tym bloku nie ma {brakuje} z {k * k} wycinków. W danych nie ma tych "
                                       "kwadratów (szare pola na wykresach), więc detekcja ich nie obejmuje.")

                H, W = mask.shape
                px_budynki = int(mask.sum())
                wazne = int(np.isfinite(data).sum()) if k > 1 else mask.size
                pokrycie = 100 * px_budynki / max(wazne, 1)

                c1, c2, c3 = st.columns(3)
                c1.metric("Wykryte piksele zabudowy", f"{px_budynki:,}".replace(",", " "))
                c2.metric("Pokrycie obszaru", f"{pokrycie:.1f}%")
                c3.metric("Rozmiar rastra", f"{W} × {H} px")
                st.write("")

                fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
                fig.patch.set_alpha(0)

                im = ax[0].imshow(data, cmap="terrain")
                ax[0].set_title("Model terenu (NMT)", fontsize=13, color=INK, pad=10)
                fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.02)

                brak_px = np.isnan(data) if k > 1 else None
                maska_pokaz = np.ma.masked_where(brak_px, mask) if k > 1 else mask
                if k > 1:
                    for a_ in ax:
                        a_.set_facecolor("#C9D6DC")      # szare pola = brak wycinka (brak danych)
                ax[1].imshow(maska_pokaz, cmap=ListedColormap([MIST, INK]))
                ax[1].set_title("Maska budynków (U-Net)", fontsize=13, color=INK, pad=10)

                ax[2].imshow(data, cmap="gray")
                ax[2].imshow(
                    np.ma.masked_where(mask == 0, mask),
                    cmap=ListedColormap([ALERT]), alpha=0.75,
                )
                ax[2].set_title("Nałożenie na teren", fontsize=13, color=INK, pad=10)

                for a_ in ax:
                    a_.axis("off")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # ---------- SYMULACJA POWODZI ----------
        else:
            if len(gify) > 1:
                rzeka = st.radio("Scenariusz", list(gify), horizontal=True,
                                 key=f"gif_{wybrane}")
            else:
                rzeka = next(iter(gify))
            _, mid, _ = st.columns([1, 4, 1])
            with mid:
                st.image(
                    gify[rzeka],
                    use_container_width=True,
                    caption=f"Symulacja zalania ({rzeka}) nałożona na wykryte budynki",
                )