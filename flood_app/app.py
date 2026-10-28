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
        background: {MIST}; border-radius: 14px; padding: 1.1rem 1.3rem;
        border-left: 5px solid {WATER}; margin-bottom: .9rem;
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
def load_poland_geojson():
    url = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries/POL.geo.json"
    try:
        return requests.get(url, timeout=5).json()
    except Exception:
        return None


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
    font-family: 'Manrope', sans-serif; font-weight: 700; font-size: 14px;
    color: {INK}; border: none; border-radius: 8px;
    padding: 5px 10px; box-shadow: 0 2px 8px rgba(0,0,0,.2);
}}
</style>
"""


def zbuduj_mape_polski():
    m = folium.Map(
        location=[52.0, 19.3],
        zoom_start=6,
        tiles=None,
        zoom_control=False,
        scrollWheelZoom=False,
        dragging=False,
        doubleClickZoom=False,
        touchZoom=False,
        boxZoom=False,
        keyboard=False,
        attributionControl=False,
    )
    m.get_root().html.add_child(folium.Element(PULSE_CSS))

    geo = load_poland_geojson()
    if geo:
        folium.GeoJson(
            geo,
            style_function=lambda _: {
                "fillColor": WATER, "color": INK,
                "weight": 1.6, "fillOpacity": 0.22,
            },
        ).add_to(m)

    for nazwa, d in MIASTA.items():
        folium.Marker(
            location=[d["lat"], d["lon"]],
            tooltip=folium.Tooltip(nazwa, direction="top", offset=(0, -10),
                                   permanent=False, class_name="city-tip"),
            icon=folium.DivIcon(
                html='<div class="pin"></div>',
                icon_size=(18, 18), icon_anchor=(9, 9),
            ),
        ).add_to(m)

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


def mapa_siatki(pozycje, wybrany):
    """Satelitarna mapa z kwadratami = wycinki 256x256 w ich prawdziwym położeniu."""
    features, lons, lats = [], [], []
    for i, rogi in enumerate(pozycje["rogi"]):
        if not rogi:
            continue
        lons += [p[0] for p in rogi]
        lats += [p[1] for p in rogi]
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
        wyb = f["properties"]["idx"] == wybrany
        return {
            "color": ALERT if wyb else "white",
            "weight": 3 if wyb else 1,
            "fillColor": ALERT if wyb else WATER,
            "fillOpacity": 0.45 if wyb else 0.08,
        }

    folium.GeoJson(
        {"type": "FeatureCollection", "features": features},
        style_function=styl,
        highlight_function=lambda f: {"weight": 3, "fillColor": WATER, "fillOpacity": 0.35},
        tooltip=folium.GeoJsonTooltip(fields=["idx"], aliases=["Wycinek:"]),
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
            height=640,
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
            st.markdown(
                f'<div class="city-card"><h3>{nazwa}{badge}</h3><p>{d["opis"]}</p></div>',
                unsafe_allow_html=True,
            )
            if st.button(f"Analizuj: {nazwa}", key=f"btn_{nazwa}", use_container_width=True):
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
                st.caption(
                    f"Kliknij kwadrat na mapie ({sum(1 for r in pozycje['rogi'] if r)} wycinków "
                    "256×256 px), aby przejść do detekcji budynków."
                )
                st.session_state.setdefault(ver_key, 0)
                map_data = st_folium(
                    mapa_siatki(pozycje, idx),
                    height=560,
                    use_container_width=True,
                    key=f"siatka_{wybrane}_{st.session_state[ver_key]}",
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
                if n > 1:
                    st.caption(f"Wybrany wycinek: {idx} z {n - 1} (256×256 px)")
                else:
                    st.caption("Plik zawiera jeden raster terenu (bez podziału na wycinki).")
                with st.spinner("Model przetwarza dane NMT…"):
                    data, mask = run_segmentation(npy_path, idx)

                H, W = mask.shape
                px_budynki = int(mask.sum())
                pokrycie = 100 * px_budynki / mask.size

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

                ax[1].imshow(mask, cmap=ListedColormap([MIST, INK]))
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
            if idx is not None and n > 1 and idx != d["patch"]:
                st.caption(f"Symulacja dotyczy wycinka {d['patch']}, a nie wybranego {idx}.")
            _, mid, _ = st.columns([1, 4, 1])
            with mid:
                st.image(
                    gify[rzeka],
                    use_container_width=True,
                    caption=f"Symulacja zalania ({rzeka}) nałożona na wykryte budynki",
                )