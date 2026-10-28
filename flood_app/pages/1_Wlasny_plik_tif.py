"""
Strona: symulacja powodzi dla WLASNEGO pliku NMPT (.tif).

Uzytkownik wgrywa GeoTIFF z modelem wysokosciowym, wybiera kadr (automatycznie na rzece
albo recznie), scenariusz wzrostu wody i dostaje animacje GIF.
Cala logika jest w flood_engine.py; tu jest tylko interfejs.

Limity (zmienne srodowiskowe):
  FLOOD_MAX_KM    najwiekszy bok kadru w km (domyslnie 1.0)
  FLOOD_MAX_JOBS  ile animacji naraz (domyslnie 1) - reszta dostaje komunikat "serwer zajety"
"""
import base64
import gc
import hashlib
import inspect
import io
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_folium import st_folium

st.set_page_config(page_title="Własny plik .tif - analiza powodziowa",
                   page_icon="🌊", layout="wide")

import flood_engine as fe
import ui_common as ui

ui.styl()
ui.wymagaj_logowania()
ui.nawigacja()
if not fe.HAS_RASTERIO:
    st.title("Własny plik .tif")
    st.error("Na serwerze nie jest zainstalowana biblioteka rasterio, więc wgrywanie plików .tif "
             "jest wyłączone. Zainstaluj ją (pip install rasterio) lub skorzystaj ze strony "
             "„Wybierz miejsce”.")
    st.stop()
MAX_KM = ui.MAX_KM
ALERT, INK = ui.ALERT, ui.INK


def zapisz(up, rodzaj):
    """Zapisuje wgrany plik na dysk (raz na plik) i zwraca sciezke."""
    klucz = f"plik_{rodzaj}"
    sig = (up.name, up.size)
    zap = st.session_state.get(klucz)
    if zap and zap["sig"] == sig and Path(zap["path"]).exists():
        return zap["path"]
    sufiks = Path(up.name).suffix.lower() or ".tif"
    sciezka = ui.katalog_sesji() / f"{rodzaj}_{time.time_ns()}{sufiks}"
    with open(sciezka, "wb") as f:
        f.write(up.getbuffer())
    st.session_state[klucz] = {"sig": sig, "path": str(sciezka)}
    return str(sciezka)


@st.cache_data(show_spinner="Przygotowuję podgląd pliku…", max_entries=4)
def podglad(sciezka, mtime):
    return fe.read_overview(sciezka)


@st.cache_data(show_spinner=False, max_entries=4)
def obraz_podgladu(sciezka, mtime):
    """Podglad calego pliku jako PNG (data URI), nakladany na mape do wyboru kadru."""
    ov, _ = podglad(sciezka, mtime)
    lo, hi = np.percentile(ov, (1, 99))
    x = np.clip((ov - lo) / max(float(hi - lo), 1e-6), 0, 1)
    rgba = (matplotlib.colormaps["terrain"](x) * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


@st.cache_data(show_spinner=False, ttl=3600)
def rozmiar_zdalny(url):
    return fe.rozmiar_zdalnego_mb(url)


# ----------------------------------------------------------------------
# Interfejs
# ----------------------------------------------------------------------
ui.sprzataj()

ui.naglowek("📂", "Symulacja z pliku .tif",
            "Wgraj model wysokościowy NMPT albo wklej link do dużego pliku, wskaż na mapie miejsce z rzeką "
            "i wybierz scenariusz. Aplikacja wykryje budynki i pokaże w animacji GIF, które z nich zostaną zalane.")
KROKI = ["Dane", "Kadr", "Scenariusz", "Wynik"]
pasek_krokow = st.empty()
pasek_krokow.markdown(ui.postep(KROKI, 0), unsafe_allow_html=True)

INSTRUKCJA_LINK = """
Dla **dużych arkuszy** (np. holenderskich, po kilkaset MB) nie trzeba nic wgrywać: wklej adres pliku w
zakładce **„Wklej link”**, a serwer pobierze tylko podgląd i wybrany kadr. Adres musi prowadzić
**wprost do pliku .tif** (nie do strony ani do archiwum .zip). Najlepiej, żeby był to plik **COG**
(Cloud Optimized GeoTIFF).

**Holandia (AHN, model powierzchni DSM 0,5 m)**

1. Otwórz spis arkuszy: https://service.pdok.nl/rws/actueel-hoogtebestand-nederland/atom/dsm_05m.xml
2. Znajdź arkusz na mapie lub liście (nazwy w rodzaju `R_26CZ2`) i kliknij go, a potem **Download**.
3. Kliknij przycisk **Download prawym przyciskiem** i wybierz **Kopiuj adres linku**. Adres wygląda tak:
   `https://service.pdok.nl/rws/ahn/atom/downloads/dsm_05m/R_26CZ2.tif`
4. Wklej go w zakładce „Wklej link”.

W AHN punkty sklasyfikowane jako woda są pomijane, więc rzeka może być dziurą w danych i wykrywanie
poziomu wody bywa tam zawodne. Sprawdź wynik na mapie.

**Szwajcaria (swissSURFACE3D Raster, DSM 0,5 m, kafelki 1 × 1 km)**

1. Otwórz https://www.swisstopo.admin.ch/en/height-model-swisssurface3d-raster i przewiń do sekcji
   **swissSURFACE3D Raster - Download**.
2. Zostaw tryb **Selection by click**, przybliż mapę do rzeki i **kliknij kafelek**, który ją obejmuje
   (siatka co 1 km, wybrany kafelek robi się zielony). Rzeka powinna być mniej więcej w jego środku.
3. Ustaw opcje: **Format: Cloud Optimized GeoTIFF**, **Resolution: 0.5**, **Coordinate system: LV95**,
   **Time status: Current**, i kliknij **Search**.
4. W wynikach pojawi się plik w rodzaju `swisssurface3d-raster_2023_2600-1201_0.5_2056_5728.tif`.
   Kliknij **Download prawym przyciskiem** i wybierz **Kopiuj adres linku**, a potem wklej go w zakładce
   „Wklej link”. Kafelek ma tylko ok. 16 MB, więc można go też pobrać i **wgrać jako plik**.
5. Kafelek ma 1 × 1 km, więc kadr obejmie cały kafelek (nie da się go przesunąć na bok). Gdy aplikacja
   zgłosi, że domena nie jest dozwolona, administrator dopisuje ją w `FLOOD_URL_HOSTS`.

**Anglia (Environment Agency, LIDAR Composite First Return DSM, 1 m, kafelki 5 × 5 km)**

Portal oddaje **archiwum ZIP**, więc tu nie działa link: pobierz plik, rozpakuj go i **wgraj** w zakładce
„Wgraj plik”.

1. Otwórz https://environment.data.gov.uk/survey (Defra Survey Data Download).
2. Przybliż mapę do rzeki i **narysuj prostokąt** na jej fragmencie (narzędzie rysowania po lewej).
3. W panelu **Download** wybierz **Select product: LIDAR Composite First Return DSM**. Nie wybieraj DTM, bo to
   model terenu bez budynków.
4. Ustaw **Select year** (najnowszy dostępny, np. 2022) i **Select resolution: 1m**.
5. W **Available tiles** pojawi się kafelek w rodzaju `lidar_composite_first_return_dsm-2022-1-NZ26se`. Kliknij jego
   nazwę albo **Download All**. Przygotowanie pliku może potrwać do kilku minut.
6. **Rozpakuj ZIP** i wgraj znajdujący się w nim plik `.tif` w zakładce „Wgraj plik”. Kafelek ma zwykle
   ok. 100 MB, czyli mieści się w limicie wgrywania.
7. Dane mają 1 m (model uczono na 0,5 m), więc budynki mogą wychodzić mniej dokładnie. Układ współrzędnych
   to EPSG:27700 (British National Grid), aplikacja odczyta go z pliku. Licencja: Open Government Licence.

**Estonia (Maa- ja Ruumiamet, DSM 1 m GeoTIFF, arkusze map; arkusz ok. 4 MB)**

1. Otwórz **Download Elevation Data**: https://geoportaal.maaruum.ee/eng/Maps-and-Data/Elevation-data/Download-Elevation-Data-p664.html
2. Ustal **numer arkusza**: w opisie obok pola wyszukiwania kliknij mapę **1:2000**, przybliż do rzeki, kliknij
   arkusz (podświetli się na czerwono) i odczytaj **Map sheet number** w oknie Object info (np. `588542`).
   Numery arkuszy 1:2000 mieszczą się między 377650 a 634632.
3. Wpisz numer w pole **Map sheet number** i kliknij **Search**. Pod polem pojawi się nagłówek
   **„Absolute height (DSM), 1m resolution GeoTIFF”** i plik w rodzaju `590551_dsm_1m.tif` (ok. 4 MB).
   Nie bierz „Laser-scanned points in LAZ format” (surowe chmury punktów) ani nDSM/DTM.
4. Kliknij plik, aby go pobrać, i **wgraj go w zakładce „Wgraj plik”** (zalecane).
   Arkusz 1:2000 to niewielki obszar, więc kadr obejmie cały arkusz.
5. Adres pliku z przeglądarki (`https://geoportaal.maaruum.ee/index.php?...f=590551_dsm_1m.tif...`) też
   można wkleić w zakładce „Wklej link”: serwer pobierze plik w całości (limit 200 MB). Jeśli pojawi się
   błąd, link wygasł i trzeba wgrać pobrany plik. Układ współrzędnych to EPSG:3301 (L-EST97), piksel 1 m.
   Dane wymagają podania źródła: „Maa- ja Ruumiamet”.

**Inne kraje: czego szukać**

- **model powierzchni (DSM)**, czyli z budynkami i drzewami, a nie model terenu (DTM),
- piksel **0,5 m** (model uczono na takim; 1 m jest powiększane, więc budynki wychodzą mniej dokładnie),
- układ współrzędnych **w metrach** (nie stopnie), format **GeoTIFF**,
- domena adresu musi być na liście dozwolonych: """ + ", ".join(fe.DOZWOLONE_HOSTY) + """
  (administrator dodaje kolejne w zmiennej `FLOOD_URL_HOSTS`).

Małe kafelki (do kilkudziesięciu MB) można po prostu **wgrać jako plik**, link jest potrzebny dopiero
przy dużych arkuszach.
"""

INSTRUKCJA_GEOPORTAL = """
Potrzebujesz plików z **tego samego obszaru**: modelu pokrycia terenu **NMPT** (wymagany) oraz,
opcjonalnie, modelu terenu **NMT** (dokładniejszy grunt, bez budynków i drzew).

**Pobieranie z geoportalu**

1. Wejdź na stronę **https://www.geoportal.gov.pl** i wybierz **Geoportal Krajowy** (pierwsza opcja,
   lewy górny róg).
2. Rozwiń **menu w lewym górnym rogu** i wybierz **Pobieranie danych**.
3. Kliknij **Pobierz dane z usługi WCS**.
4. W polu **Wybierz usługę** wybierz **Numeryczny Model Pokrycia Terenu**.
5. Upewnij się, że warstwa to **DSM_PL-KRON86-NH**.
6. Kliknij **Rysuj obszar** i zaznacz na mapie interesujący Cię fragment (wskazówki poniżej).
7. Kliknij **Pobierz plik**. Dostaniesz plik **.asc** - to jest NMPT.
8. **Nie usuwaj zaznaczonego obszaru.** W polu **Usługa WCS** wybierz teraz
   **Numeryczny Model Terenu - GeoTIFF** i ponownie kliknij **Pobierz plik**.
   Dostaniesz plik **.tiff** - to jest NMT (krok opcjonalny).
9. Wgraj NMPT w pierwsze pole poniżej, a NMT (jeśli go masz) w trzecie. Pole z maską budynków
   możesz zostawić puste - budynki wykryje model.

**Jak zaznaczyć obszar**

- Najpierw **oddal mapę**, aż zobaczysz rzekę z zabudową po obu brzegach, i dopiero wtedy rysuj.
  Przy dużym przybliżeniu prostokąt wygląda na duży, a w rzeczywistości jest malutki - zerknij na
  etykietę z powierzchnią.
- Dobry rozmiar to **250 000 - 1 000 000 m²** (od 500 × 500 m do 1 × 1 km). Poniżej 128 m z jednej
  strony plik zostanie odrzucony.
- **Rzeka powinna być w środku obszaru** i widoczna jako woda, bo poziom wody liczymy od jej
  powierzchni.
- Nie zaznaczaj więcej niż potrzeba: plik .asc zajmuje ok. 20 bajtów na piksel, czyli
  1 km² to ok. 80 MB, a limit wgrywania to 300 MB (ok. 3 km²).

**Najczęstsze problemy**

- *„Plik jest za mały"* - zaznacz większy obszar.
- *„Plik .asc nie zawiera układu współrzędnych"* - to normalne, strona zakłada EPSG:2180.
- *„Nie użyto NMT: ..."* - NMT musi obejmować ten sam (lub większy) obszar co NMPT i być w tej samej
  wersji układu wysokości (KRON86). Pobierz oba pliki bez odznaczania obszaru. Animacja i tak
  powstanie, tylko z gruntem oszacowanym z NMPT.
- *„W kadrze nie wykryto rzeki"* - zaznacz obszar, na którym jest rzeka.
"""

ui.krok(1, "Dane wysokościowe", "Plik NMPT z geoportalu (GeoTIFF lub .asc) albo adres dużego arkusza, np. holenderskiego AHN.")
with ui.karta("dane"):
    zakl_plik, zakl_link = st.tabs(["📂 Wgraj plik", "🔗 Wklej link (duże pliki)"])
    with zakl_plik:
        nmpt_up = st.file_uploader("Model wysokościowy NMPT (GeoTIFF lub .asc)", type=["tif", "tiff", "asc"],
                                   key="up_nmpt",
                                   help="Surowy arkusz z geoportalu (Arc/Info GRID, .asc) też zadziała.")
        with st.expander("Dodatkowe pliki (opcjonalnie): maska budynków i model terenu NMT"):
            kb, kc = st.columns(2)
            maska_up = kb.file_uploader("Maska budynków", type=["tif", "tiff"], key="up_maska",
                                        help="Raster 0/1 w tym samym układzie współrzędnych co NMPT. "
                                             "Bez niej budynki wykryje model U-Net.")
            nmt_up = kc.file_uploader("Model terenu NMT", type=["tif", "tiff", "asc"], key="up_nmt",
                                      help="Grunt bez budynków i drzew. Pobierz dla tego samego lub nieco większego "
                                           "obszaru i w tym samym układzie wysokości (najlepiej KRON86 jak NMPT). "
                                           "Bez niego grunt jest szacowany z NMPT.")
    with zakl_link:
        url_nmpt = st.text_input(
            "Adres (https) pliku GeoTIFF / COG", key="url_nmpt", placeholder="https://service.pdok.nl/…/R_26CZ2.tif",
            help="Plik Cloud Optimized GeoTIFF jest czytany fragmentami: serwer pobiera tylko podgląd i wybrany "
                 "kadr, a nie cały arkusz (np. 335 MB). Na stronie z danymi kliknij prawym przyciskiem przycisk "
                 "Download i skopiuj adres linku. Dozwolone domeny: " + ", ".join(fe.DOZWOLONE_HOSTY) + ".").strip()
        st.caption("Nic nie jest wgrywane: serwer czyta z linku tylko podgląd i wybrany kadr. "
                   "Dozwolone domeny: " + ", ".join(fe.DOZWOLONE_HOSTY) + ".")
    with st.expander("Jak pobrać dane z geoportalu (krok po kroku)"):
        st.markdown(INSTRUKCJA_GEOPORTAL)

    with st.expander("Dane spoza Polski i linki do dużych plików (Holandia i inne)"):
        st.markdown(INSTRUKCJA_LINK)
    with st.expander("Wymagania dla pliku"):
        st.markdown(
            "- **Układ współrzędnych:** w metrach, np. EPSG:2180. Plik w stopniach (EPSG:4326) nie zadziała.\n"
            "- **Rozdzielczość:** najlepiej 0,5 m na piksel (na takiej uczono model).\n"
            "- **Rozmiar:** limit wgrywania to 300 MB. Większy plik (np. arkusz AHN z Holandii, 335 MB) wskaż "
            "**adresem https** (pole pod wgrywaniem): jeśli to COG, serwer pobierze tylko podgląd i wybrany "
            "kadr. Można też przyciąć plik wcześniej, np. w QGIS.\n"
            "- **Maska budynków** jest opcjonalna. Bez niej budynki wykryje model U-Net."
        )

if nmpt_up is None and not url_nmpt:
    st.info("Wgraj plik NMPT albo wklej adres pliku, aby zacząć.")
    st.stop()

if nmpt_up is not None:
    sciezka = zapisz(nmpt_up, "nmpt")
    zrodlo_sig, zrodlo_mtime = (nmpt_up.name, nmpt_up.size), os.path.getmtime(sciezka)
else:
    sciezka, blad_url = fe.url_do_sciezki(url_nmpt)
    if blad_url:
        st.error(blad_url)
        st.stop()
    zrodlo_sig, zrodlo_mtime = (url_nmpt, 0), 0
try:
    info, bledy, ostrz = fe.inspect_tif(sciezka)
except Exception as exc:
    st.error(f"Nie mogę odczytać pliku jako GeoTIFF: {exc}")
    st.stop()

pasek_krokow.markdown(ui.postep(KROKI, 3), unsafe_allow_html=True)
ui.chipy([("Rozmiar", f"{info['width']} × {info['height']} px"), ("Piksel", f"{info['res'][0]:.2f} m"),
          ("Układ", info["crs"] or "brak"),
          ("Źródło", f"plik {info['size_mb']:.0f} MB" if info["size_mb"] is not None else "link (czytany fragmentami)")])

for b in bledy:
    st.error(b)
if bledy:
    st.stop()
for w in ostrz:
    st.warning(w)

maska_sciezka = None
if maska_up is not None:
    maska_sciezka = zapisz(maska_up, "maska")
    try:
        info_m, bledy_m, _ = fe.inspect_tif(maska_sciezka)
        if bledy_m:
            st.error("Maska: " + bledy_m[0])
            st.stop()
        if not fe.same_crs(info_m, info):
            st.error(f"Maska jest w innym układzie współrzędnych ({info_m['crs']}) niż NMPT "
                     f"({info['crs']}). Zapisz oba pliki w tym samym układzie.")
            st.stop()
    except Exception as exc:
        st.error(f"Nie mogę odczytać maski: {exc}")
        st.stop()

nmt_sciezka = None
if nmt_up is not None:
    nmt_sciezka = zapisz(nmt_up, "nmt")
    try:
        info_n, bledy_n, ostrz_n = fe.inspect_tif(nmt_sciezka)
        bledy_n = [b for b in bledy_n if "za mały" not in b]      # NMT moze byc wiekszy lub mniejszy niz kadr
        if bledy_n:
            st.error("NMT: " + bledy_n[0])
            st.stop()
        if not fe.same_crs(info_n, info):
            st.error(f"NMT jest w innym układzie współrzędnych ({info_n['crs']}) niż NMPT ({info['crs']}).")
            st.stop()
    except Exception as exc:
        st.error(f"Nie mogę odczytać NMT: {exc}")
        st.stop()

# ---- 1. kadr ----
ui.krok(2, "Kadr", "Kliknij na mapie miejsce z rzeką i zabudową. Czerwony kwadrat to obszar, który zostanie przeliczony.")
ov, (sx, sy) = podglad(sciezka, zrodlo_mtime)
H, W = ov.shape
rozmiar_m = min(info["width"] * info["res"][0], info["height"] * info["res"][1])
km_max = min(MAX_KM, rozmiar_m / 1000)

if st.session_state.get("kadr_zrodlo") != zrodlo_sig:           # nowy plik: wracamy do kadru na rzece
    st.session_state.update(kadr_zrodlo=zrodlo_sig, kadr_srodek=None, kadr_klik=None)

c1, c2 = st.columns([1, 2], gap="large")
with c1:
    with ui.karta("kadr"):
        krok = 0.25
        max_slider = int(km_max / krok + 1e-9) * krok      # zaokraglenie W DOL do kroku
        zrodlo_limitu = (f"obliczony z pamięci serwera ({ui.pamiec_calkowita_mb() or 0:.0f} MB RAM)"
                         if ui.MAX_KM_AUTO else "ustawiony przez administratora (FLOOD_MAX_KM)")
        if max_slider > 0.5:
            km = st.slider("Rozmiar kadru (km)", 0.5, float(max_slider),
                           float(min(1.0, max_slider)), krok,
                           help=f"Większy kadr = dłuższe liczenie i więcej pamięci. Limit serwera: {MAX_KM:g} km, "
                                f"{zrodlo_limitu}.")
        else:
            km = float(km_max)
            st.caption(f"Plik jest mały: kadr obejmie cały obszar ({km:.2f} km).")

        plik_km2 = info["width"] * info["res"][0] * info["height"] * info["res"][1] / 1e6
        udzial = km * km / plik_km2 if plik_km2 > 0 else 1.0
        linia = f"Kadr {km:g} × {km:g} km to **{udzial * 100:.0f}%** powierzchni pliku ({plik_km2:.1f} km²)."
        if info["size_mb"] is None:                                   # plik zdalny (z linku)
            mb = rozmiar_zdalny(url_nmpt)
            if mb:
                linia += f" Z linku pobierzemy tylko ten fragment: ok. **{mb * min(udzial, 1):.0f} MB** z {mb:.0f} MB."
            else:
                linia += " Z linku pobierzemy tylko ten fragment, a nie cały plik."
        else:
            linia += " Plik jest już na serwerze, liczony będzie tylko ten kadr."
        st.markdown(linia)

        if st.button("Wyśrodkuj na rzece", disabled=st.session_state["kadr_srodek"] is None):
            st.session_state["kadr_srodek"] = None
            st.rerun()
        st.caption("Położenie kadru: " + ("ustawione ręcznie na mapie." if st.session_state["kadr_srodek"]
                                           else "automatycznie na rzece z zabudową. Kliknij na mapie, aby je zmienić."))

win_ov = km * 1000 / info["res"][0] / sx          # bok kadru w pikselach podgladu
if st.session_state["kadr_srodek"] is None:
    cx_ov, cy_ov = fe.auto_center(ov, win_ov)
else:
    cx_ov, cy_ov = st.session_state["kadr_srodek"]

win_ov = min(win_ov, W, H)
x0 = float(np.clip(cx_ov - win_ov / 2, 0, W - win_ov))
y0 = float(np.clip(cy_ov - win_ov / 2, 0, H - win_ov))
cx_src, cy_src = (x0 + win_ov / 2) * sx, (y0 + win_ov / 2) * sy

with c2:
    # Mapa w ukladzie "plaskim" (CRS Simple): lat = y rosnacy w gore, lng = x. Wiersz podgladu r -> lat = H - r.
    HAS_FG = "feature_group_to_add" in inspect.signature(st_folium).parameters
    klucz_mapy = "mapa_kadr_" + hashlib.md5(repr(zrodlo_sig).encode()).hexdigest()[:10]
    widok = None if HAS_FG else st.session_state.get("widok_" + klucz_mapy)   # starsza wersja: sami pamietamy widok
    m = folium.Map(location=widok["center"] if widok else [H / 2, W / 2],
                   zoom_start=widok["zoom"] if widok else 0, crs="Simple", tiles=None,
                   min_zoom=-3, max_zoom=4, zoom_snap=0.25, attributionControl=False)
    folium.raster_layers.ImageOverlay(image=obraz_podgladu(sciezka, zrodlo_mtime),
                                      bounds=[[0, 0], [H, W]]).add_to(m)
    if not widok:
        m.fit_bounds([[0, 0], [H, W]])
    # Nie pozwalamy wyjechac poza obraz: widok zostaje w granicach pliku, nie da sie oddalic bardziej,
    # niz mieści sie caly plik, a tlo ma kolor strony zamiast szarosci.
    mj = m.get_name()
    m.get_root().header.add_child(folium.Element("<style>.leaflet-container{background:#F6F9FA !important;}</style>"))
    # Skrypt musi poczekac na zdarzenie load: folium dopisuje tworzenie mapy PO naszych elementach,
    # wiec wykonany od razu odwolalby sie do jeszcze nieistniejacej mapy i zepsul cala mape.
    m.get_root().script.add_child(folium.Element(
        f"window.addEventListener('load', function() {{ {mj}.options.maxBoundsViscosity = 1.0; "
        f"{mj}.setMaxBounds([[0, 0], [{H}, {W}]]); "
        f"{mj}.setMinZoom({mj}.getBoundsZoom([[0, 0], [{H}, {W}]])); }});"))
    fg = folium.FeatureGroup(name="kadr")
    folium.Rectangle([[H - (y0 + win_ov), x0], [H - y0, x0 + win_ov]], color=ALERT, weight=3,
                     fill=True, fill_opacity=0.12).add_to(fg)
    kwargs = {"returned_objects": ["last_clicked"] if HAS_FG else ["last_clicked", "center", "zoom"]}
    if HAS_FG:
        kwargs["feature_group_to_add"] = fg          # kwadrat dodawany bez resetowania widoku mapy
    else:
        fg.add_to(m)
    wyj = st_folium(m, height=560, use_container_width=True, key=klucz_mapy, **kwargs) or {}
    if not HAS_FG and wyj.get("center") and wyj.get("zoom") is not None:
        st.session_state["widok_" + klucz_mapy] = {"center": [wyj["center"]["lat"], wyj["center"]["lng"]],
                                                   "zoom": wyj["zoom"]}
    st.caption("Kliknij w miejsce, które Cię interesuje (np. rzeka z zabudową): tam ustawi się środek kadru. "
               "Przybliżanie: kółko myszy.")

klik = wyj.get("last_clicked")
if klik:
    nowy_klik = (round(klik["lat"], 4), round(klik["lng"], 4))
    if nowy_klik != st.session_state["kadr_klik"]:
        st.session_state["kadr_klik"] = nowy_klik
        st.session_state["kadr_srodek"] = (float(np.clip(klik["lng"], 0, W)), float(np.clip(H - klik["lat"], 0, H)))
        st.rerun()

# ---- 2. scenariusz ----
ui.krok(3, "Scenariusz wzrostu wody", "O ile metrów podnosimy poziom względem rzeki widocznej w NMPT.")
with ui.karta("scenariusz"):
    max_eff, krok = ui.kontrolki_scenariusza()

zrodlo_maski = "z wgranego pliku" if maska_sciezka else "model U-Net (best_model.pth)"
if not maska_sciezka and not ui.model_dostepny():
    st.error("Brak pliku best_model.pth na serwerze, więc nie da się wykryć budynków. "
             "Wgraj maskę budynków albo skopiuj model do folderu aplikacji.")
    st.stop()

# ---- 3. generowanie ----
ui.krok(4, "Generuj animację")
st.caption(f"Maska budynków: {zrodlo_maski}. Generowanie trwa zwykle kilka minut i zależy od "
           "rozmiaru kadru. Nie zamykaj karty.")

sig_biezacy = (*zrodlo_sig, round(cx_src), round(cy_src), km, max_eff, bool(maska_sciezka),
               bool(nmt_sciezka))

if st.button("Generuj GIF", type="primary"):
    sem, blad = ui.zarezerwuj(km, not maska_sciezka)
    if blad:
        st.error(blad) if "pamięci" in blad else st.warning(blad)
    else:
        t0 = time.time()
        pasek = st.progress(0.0, text="Start…")
        try:
            predykcja = None
            if not maska_sciezka:
                model, device = ui.zaladuj_model()
                predykcja = ui.funkcja_predykcji(model, device)
            wyjscie = ui.katalog_sesji() / f"powodz_{int(t0)}.gif"
            wynik = fe.generate_gif(
                sciezka, str(wyjscie), cx_src, cy_src, crop_km=km,
                mask_path=maska_sciezka, predict_prob=predykcja, nmt_path=nmt_sciezka,
                max_rise_m=max_eff, step_m=krok, fps=4,
                progress=lambda msg, f: pasek.progress(f, text=msg),
            )
            st.session_state["wynik"] = {"path": str(wyjscie), "info": wynik,
                                         "t": time.time() - t0, "sig": sig_biezacy}
            pasek.empty()
        except Exception as exc:
            pasek.empty()
            st.error(f"Nie udało się wygenerować animacji: {exc}")
        finally:
            sem.release()
            gc.collect()

wyn_biezacy = st.session_state.get("wynik")
if wyn_biezacy and wyn_biezacy["sig"] == sig_biezacy:
    pasek_krokow.markdown(ui.postep(KROKI, 4), unsafe_allow_html=True)
ui.pokaz_wynik(wyn_biezacy, sig_biezacy)