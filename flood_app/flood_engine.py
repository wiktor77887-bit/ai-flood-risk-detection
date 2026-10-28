"""
flood_engine.py - silnik generowania GIF-a z DOWOLNEGO wgranego pliku NMPT (.tif)

Odpowiada za wszystko, co nie jest interfejsem:
  * walidacja pliku (uklad w metrach, rozdzielczosc, typ danych),
  * podglad calego pliku w malej rozdzielczosci (bez wczytywania gigabajtow do RAM-u),
  * automatyczne znalezienie kadru z rzeka,
  * wczytanie TYLKO wybranego kadru (odczyt okna),
  * maska budynkow: z wgranego pliku albo z modelu U-Net (kafelkami),
  * zdarzenia powodzi wzgledem stanu normalnego i wywolanie animacja_miasto.make_animation.

Interfejs Streamlit jest w pages/1_Wlasny_plik_tif.py.
"""
import os
from urllib.parse import urlparse

import numpy as np

# Odczyt zdalnych plikow COG (Cloud Optimized GeoTIFF) przez HTTP: GDAL pobiera tylko potrzebne
# fragmenty (podglad z wbudowanych poziomow, potem tylko wybrany kadr), a nie caly plik.
for _k, _v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
               "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
               "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
               "GDAL_HTTP_TIMEOUT": "60",
               "GDAL_HTTP_MAX_RETRY": "3",
               "VSI_CACHE": "TRUE"}.items():
    os.environ.setdefault(_k, _v)

# Z jakich domen wolno czytac pliki po adresie URL (ochrona przed zapytaniami do dowolnych adresow).
# Mozna rozszerzyc zmienna srodowiskowa, np. FLOOD_URL_HOSTS="pdok.nl,data.geo.admin.ch,moja-domena.pl".
DOZWOLONE_HOSTY = tuple(h.strip().lower().lstrip(".") for h in
                        os.environ.get("FLOOD_URL_HOSTS", "pdok.nl,data.geo.admin.ch,geoportaal.maaruum.ee").split(",") if h.strip())
# Domeny, ktore oddaja plik przez skrypt (adres nie konczy sie na .tif, brak odczytu fragmentami):
# serwer pobiera taki plik w calosci (limit ponizej) do pamieci podrecznej na dysku.
POBIERANE_HOSTY = ("maaruum.ee",)
MAX_POBRANIE_MB = float(os.environ.get("FLOOD_MAX_DOWNLOAD_MB", "200"))
try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import Window, from_bounds
    HAS_RASTERIO = True
except ImportError:          # bez rasterio dziala tylko generowanie z gotowej tablicy (strona z mapa)
    rasterio = Resampling = Window = from_bounds = None
    HAS_RASTERIO = False
from scipy.ndimage import uniform_filter

import animacja_miasto as am

TARGET_RES = 0.5     # model byl trenowany na pikselu 0.5 m
PALETTE = ["#2196F3", "#FF9800", "#FF5722", "#E91E63", "#9C27B0",
           "#673AB7", "#1A237E", "#0D47A1", "#4A148C", "#212121"]
NORMAL_CM = 200      # umowny "stan normalny" (tylko punkt odniesienia)


# ----------------------------------------------------------------------
# Walidacja
# ----------------------------------------------------------------------
def _need_rasterio():
    if not HAS_RASTERIO:
        raise RuntimeError("Na serwerze nie ma biblioteki rasterio, więc odczyt plików .tif jest wyłączony.")


def url_do_sciezki(url):
    """Adres https do pliku GeoTIFF/COG -> (sciezka dla GDAL, None) albo (None, komunikat bledu)."""
    url = (url or "").strip()
    u = urlparse(url)
    host = (u.hostname or "").lower()
    if u.scheme != "https" or not host:
        return None, "Podaj pełny adres zaczynający się od https://"
    if u.username or u.password or (u.port not in (None, 443)):
        return None, "Adres nie może zawierać loginu ani niestandardowego portu."
    if not any(host == h or host.endswith("." + h) for h in DOZWOLONE_HOSTY):
        return None, ("Ta domena nie jest na liście dozwolonych: " + ", ".join(DOZWOLONE_HOSTY) +
                      ". Jeśli pobierasz dane z innego portalu (np. angielskiego), pobierz plik .tif "
                      "(lub rozpakuj ZIP) i wgraj go w zakładce „Wgraj plik”. Administrator może też dodać "
                      "domenę w zmiennej FLOOD_URL_HOSTS.")
    if any(host == h or host.endswith("." + h) for h in POBIERANE_HOSTY):
        if ".tif" not in (u.path + "?" + u.query).lower():
            return None, "Adres musi wskazywać plik .tif."
        return _pobierz_do_cache(url)
    if not u.path.lower().endswith((".tif", ".tiff")):
        return None, "Adres musi wskazywać plik .tif."
    return "/vsicurl/" + url, None


def _pobierz_do_cache(url):
    """Pobiera caly plik (do MAX_POBRANIE_MB) do katalogu tymczasowego -> (sciezka, None) / (None, blad)."""
    import hashlib
    import tempfile
    import urllib.error
    import urllib.request

    katalog = os.path.join(tempfile.gettempdir(), "flood_url_cache")
    os.makedirs(katalog, exist_ok=True)
    cel = os.path.join(katalog, hashlib.sha1(url.encode()).hexdigest() + ".tif")
    if os.path.exists(cel) and os.path.getsize(cel) > 0:
        return cel, None
    limit = int(MAX_POBRANIE_MB * 1e6)
    czesc = cel + ".part"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "flood-app"}),
                                    timeout=60) as r, open(czesc, "wb") as f:
            dl = r.headers.get("Content-Length")
            if dl and int(dl) > limit:
                return None, f"Plik jest większy niż {MAX_POBRANIE_MB:.0f} MB. Pobierz go i wgraj ręcznie."
            wczytano = 0
            while True:
                blok = r.read(1 << 20)
                if not blok:
                    break
                wczytano += len(blok)
                if wczytano > limit:
                    return None, f"Plik jest większy niż {MAX_POBRANIE_MB:.0f} MB. Pobierz go i wgraj ręcznie."
                f.write(blok)
        with open(czesc, "rb") as f:
            if f.read(4) not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
                return None, ("Pod tym adresem nie ma pliku GeoTIFF (link mógł wygasnąć). "
                              "Pobierz plik w przeglądarce i wgraj go w zakładce „Wgraj plik”.")
        os.replace(czesc, cel)
        return cel, None
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return None, f"Nie udało się pobrać pliku z linku ({exc}). Pobierz go w przeglądarce i wgraj ręcznie."
    finally:
        if os.path.exists(czesc):
            os.remove(czesc)


def rozmiar_zdalnego_mb(url):
    """Rozmiar pliku pod adresem https (naglowek Content-Length, bez pobierania i bez podazania za
    przekierowaniami); None, gdy serwer go nie podaje."""
    import urllib.error
    import urllib.request

    class _BezPrzekierowan(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    try:
        zadanie = urllib.request.Request(url, method="HEAD")
        with urllib.request.build_opener(_BezPrzekierowan).open(zadanie, timeout=10) as r:
            dl = r.headers.get("Content-Length")
            return int(dl) / 1e6 if dl else None
    except (urllib.error.URLError, OSError, ValueError):
        return None


def inspect_tif(path):
    """Zwraca (info, bledy, ostrzezenia)."""
    _need_rasterio()
    with rasterio.open(path) as src:
        crs = src.crs
        driver = src.driver
        info = {
            "width": src.width, "height": src.height, "count": src.count,
            "dtype": src.dtypes[0],
            "crs": crs.to_string() if crs else None,
            "epsg": (crs.to_epsg() if crs else None),
            "projected": bool(crs and crs.is_projected),
            "res": (abs(src.res[0]), abs(src.res[1])),
            "bounds": tuple(src.bounds),
            "nodata": src.nodata,
            "size_mb": os.path.getsize(path) / 1e6 if os.path.exists(path) else None,   # None = plik zdalny
            "crs_assumed": False,
        }
    errors, warnings = [], []
    if crs is None and driver == "AAIGrid":
        # surowe arkusze z geoportalu (Arc/Info ASCII Grid) nie niosa ukladu wspolrzednych;
        # dane PZGiK sa w ukladzie PL-1992 (EPSG:2180), wiec zakladamy metry i ten uklad
        info["crs"], info["projected"], info["crs_assumed"], info["epsg"] = "EPSG:2180", True, True, 2180
        warnings.append("Plik .asc nie zawiera układu współrzędnych - zakładam EPSG:2180 (PL-1992), "
                        "w którym geoportal udostępnia dane.")
    if info["count"] >= 3 and info["dtype"] == "uint8":
        errors.append("To wygląda na zdjęcie (RGB), a nie na model wysokościowy NMPT.")
    if not info["projected"]:
        errors.append("Plik nie ma układu współrzędnych w metrach (np. EPSG:2180), "
                      "więc nie da się ustalić skali terenu.")
    r = info["res"][0]
    if info["projected"]:
        if r > 1.5:
            errors.append(f"Rozdzielczość {r:.2f} m/piksel jest za mała na budynki "
                          "(model wymaga ok. 0,5 m).")
        elif r > 0.75:
            warnings.append(f"Rozdzielczość {r:.2f} m/piksel: plik zostanie powiększony do 0,5 m, "
                            "detale budynków będą rozmyte.")
        elif r < 0.25:
            warnings.append(f"Rozdzielczość {r:.2f} m/piksel jest bardzo duża; "
                            "plik zostanie zmniejszony do 0,5 m.")
    if info["projected"]:
        w_m, h_m = info["width"] * info["res"][0], info["height"] * info["res"][1]
        if min(w_m, h_m) < 128:
            errors.append(f"Plik jest za mały: obejmuje {w_m:.0f} × {h_m:.0f} m, a potrzeba co najmniej "
                          "128 m w każdym kierunku. Pobierz większy obszar: najlepiej 500 × 500 m lub więcej, "
                          "z rzeką w środku i zabudową po obu brzegach.")
        elif min(w_m, h_m) < 300:
            warnings.append(f"Plik jest mały ({w_m:.0f} × {h_m:.0f} m). Zalecane jest co najmniej 500 × 500 m, "
                            "żeby w kadrze była rzeka i trochę zabudowy.")
    if info["dtype"] == "uint8":
        warnings.append("Typ danych uint8: wysokości w NMPT zwykle są zmiennoprzecinkowe.")
    return info, errors, warnings


def same_crs(info_a, info_b):
    """Czy dwa pliki sa w tym samym ukladzie (po kodzie EPSG, a gdy go brak - po opisie).
    Gdy ktorys plik nie ma ukladu (.asc, zalozony EPSG:2180), nie ma z czym porownywac - uznajemy za zgodne."""
    if info_a.get("crs_assumed") or info_b.get("crs_assumed"):
        return True
    if info_a.get("epsg") and info_b.get("epsg"):
        return info_a["epsg"] == info_b["epsg"]
    return info_a.get("crs") == info_b.get("crs")


def clean_terrain(a, nodata=None):
    """NaN dla brakow danych, potem wypelnienie srednia (jak w precompute.py)."""
    a = np.ma.filled(a.astype(np.float32), np.nan) if np.ma.isMaskedArray(a) else a.astype(np.float32)
    a[a < -100] = np.nan
    if nodata is not None:
        a[a == nodata] = np.nan
    if not np.isfinite(a).any():
        raise ValueError("Wybrany obszar nie zawiera żadnych danych wysokościowych.")
    return np.nan_to_num(a, nan=float(np.nanmean(a)))


# ----------------------------------------------------------------------
# Podglad calego pliku i wybor kadru
# ----------------------------------------------------------------------
def read_overview(path, max_px=900):
    """Maly podglad calego rastra. Zwraca (tablica, (skala_x, skala_y)) - ile px zrodla na px podgladu."""
    _need_rasterio()
    with rasterio.open(path) as src:
        scale = max(src.width, src.height) / max_px
        scale = max(scale, 1.0)
        oh = max(1, int(round(src.height / scale)))
        ow = max(1, int(round(src.width / scale)))
        a = src.read(1, out_shape=(oh, ow), resampling=Resampling.average, masked=True)
        nod = src.nodata
        sx, sy = src.width / ow, src.height / oh
    return clean_terrain(a, nod), (sx, sy)


def auto_center(ov, win_ov):
    """
    Srodek kadru (w pikselach podgladu) z rzeka i zabudowa. Heurystyka na podgladzie:
    okno, w ktorym ok. 25% to woda/niziny przy rzece i duzo "chropowatosci" (budynki, drzewa).
    """
    h, w = ov.shape
    k = int(max(2, min(round(win_ov), h, w)))
    surface = am.estimate_river_surface(ov)
    river = (ov < surface + 0.6).astype(np.float64)
    rough = (np.abs(ov - uniform_filter(ov, 5)) > 1.0).astype(np.float64)

    def wsum(a):
        I = np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return (I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]) / (k * k)

    score = np.minimum(wsum(river), 0.25) * (0.1 + wsum(rough))
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    return int(ix + k / 2), int(iy + k / 2)


def make_window(info, cx_src, cy_src, side_m):
    """Okno w pikselach zrodla (kwadrat o boku side_m metrow) wokol srodka, przesuniete do wnetrza rastra."""
    rx, ry = info["res"]
    wpx, hpx = side_m / rx, side_m / ry
    wpx, hpx = min(wpx, info["width"]), min(hpx, info["height"])
    x0 = float(np.clip(cx_src - wpx / 2, 0, info["width"] - wpx))
    y0 = float(np.clip(cy_src - hpx / 2, 0, info["height"] - hpx))
    return Window(round(x0), round(y0), round(wpx), round(hpx))


# ----------------------------------------------------------------------
# Wczytanie kadru
# ----------------------------------------------------------------------
def read_terrain_crop(path, window, info):
    """Kadr NMPT przeskalowany do 0.5 m/px. Zwraca (teren, granice_geograficzne)."""
    _need_rasterio()
    oh = int(round(window.height * info["res"][1] / TARGET_RES))
    ow = int(round(window.width * info["res"][0] / TARGET_RES))
    with rasterio.open(path) as src:
        a = src.read(1, window=window, out_shape=(oh, ow),
                     resampling=Resampling.bilinear, masked=True)
        bounds = rasterio.windows.bounds(window, src.transform)
        nod = src.nodata
    return clean_terrain(a, nod), bounds


def read_mask_crop(mask_path, bounds, out_shape):
    """Maska budynkow (0/1) dla tych samych granic geograficznych, niezaleznie od siatki pliku maski."""
    _need_rasterio()
    with rasterio.open(mask_path) as src:
        win = from_bounds(*bounds, transform=src.transform)
        m = src.read(1, window=win, out_shape=out_shape, boundless=True,
                     fill_value=0, resampling=Resampling.nearest).astype(np.float32)
        nod = src.nodata
    m[m == 255] = 0
    if nod is not None:
        m[m == nod] = 0
    return (m > 0).astype(np.float32)


def read_ground_crop(nmt_path, bounds, out_shape):
    """
    Oficjalny NMT (grunt bez budynkow i drzew) dla tych samych granic geograficznych co kadr NMPT,
    przeskalowany do jego siatki. Zwraca (tablica z NaN, ulamek kadru pokryty plikiem NMT).
    """
    _need_rasterio()
    with rasterio.open(nmt_path) as src:
        sb, nod = src.bounds, src.nodata
        left, bottom, right, top = bounds
        ow = max(0.0, min(right, sb.right) - max(left, sb.left))
        oh = max(0.0, min(top, sb.top) - max(bottom, sb.bottom))
        cover = (ow * oh) / max((right - left) * (top - bottom), 1e-9)
        win = from_bounds(*bounds, transform=src.transform)
        a = src.read(1, window=win, out_shape=out_shape, boundless=True,
                     resampling=Resampling.bilinear, masked=True)
    a = np.ma.filled(a.astype(np.float32), np.nan)
    a[a < -100] = np.nan
    if nod is not None:
        a[a == nod] = np.nan
    return a, float(cover)


def prepare_ground(terrain, nmt_crop, cover):
    """Sprawdza i uzupelnia NMT; zwraca grunt albo zglasza ValueError z powodem odrzucenia."""
    if cover < 0.95:
        raise ValueError(f"plik NMT obejmuje tylko {cover * 100:.0f}% wybranego kadru "
                         "(pobierz NMT dla tego samego lub nieco większego obszaru)")
    nan = np.isnan(nmt_crop)
    if nan.mean() > 0.2:
        raise ValueError(f"w NMT brakuje danych dla {nan.mean() * 100:.0f}% kadru")
    if nan.any():                                   # drobne luki: najblizsza wartosc
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(nan, return_distances=False, return_indices=True)
        nmt_crop = nmt_crop[tuple(idx)]
    if float(np.percentile(terrain - nmt_crop, 10)) < -2.0:
        raise ValueError("wysokości NMT nie pasują do NMPT (NMPT leży wyraźnie poniżej gruntu)")
    return nmt_crop.astype(np.float32)


def predict_mask(terrain, predict_prob, tile=512, pad=32, progress=None):
    """
    Maska budynkow z modelu, kafelkami z zakladka (pad) dla kontekstu na brzegach.
    predict_prob(tablica 2D w zakresie 0-1) -> prawdopodobienstwa tego samego ksztaltu.
    Normalizacja jak w precompute.py: percentyle 2-98 (tu: dla kadru).
    """
    p2, p98 = np.percentile(terrain, (2, 98))
    norm = np.clip((terrain - p2) / (p98 - p2 + 1e-6), 0, 1).astype(np.float32)
    H, W = norm.shape
    padded = np.pad(norm, pad, mode="reflect")
    mask = np.zeros((H, W), dtype=np.float32)
    ny, nx = -(-H // tile), -(-W // tile)
    done = 0
    for y in range(0, H, tile):
        for x in range(0, W, tile):
            y1, x1 = min(y + tile, H), min(x + tile, W)
            sub = padded[y:y1 + 2 * pad, x:x1 + 2 * pad]
            prob = predict_prob(sub)
            mask[y:y1, x:x1] = prob[pad:pad + (y1 - y), pad:pad + (x1 - x)] > 0.5
            done += 1
            if progress:
                progress(done, ny * nx)
    return mask


# ----------------------------------------------------------------------
# Scenariusze
# ----------------------------------------------------------------------
def build_events(max_rise_m=4.0, step_m=1.0):
    """Stan normalny + wzrosty co step_m do max_rise_m (maks. 10 zdarzen)."""
    events = [{"name": "Stan normalny", "gauge_cm": NORMAL_CM, "year": None,
               "color": PALETTE[0], "hide_cm": True}]
    r, k = step_m, 1
    while r <= max_rise_m + 1e-9 and k < len(PALETTE):
        events.append({"name": f"Wzrost +{r:g} m", "gauge_cm": int(round(NORMAL_CM + r * 100)),
                       "year": None, "color": PALETTE[k], "hide_cm": True})
        r += step_m
        k += 1
    return events


# ----------------------------------------------------------------------
# Calosc
# ----------------------------------------------------------------------
def generate_from_terrain(terrain, out_path, mask=None, predict_prob=None,
                          max_rise_m=4.0, step_m=1.0, fps=4, progress=None, title=None,
                          ground=None, ground_mode="estimate"):
    """
    Rdzen generowania, niezalezny od zrodla danych: dostaje gotowy kadr NMPT (0.5 m/px, metry n.p.m.)
    i opcjonalnie maske budynkow (0/1, ten sam ksztalt). Bez maski uzywa predict_prob (model).
    ground: opcjonalny model GRUNTU (np. oficjalny NMT) o tym samym ksztalcie; bez niego, przy
    ground_mode="estimate", grunt jest szacowany z NMPT (usuniete budynki i drzewa), a woda liczona
    na gruncie. ground_mode="none" - woda liczona wprost na NMPT (stara metoda, z dachami).
    Zwraca slownik z informacjami; progress(komunikat, ulamek 0-1).
    """
    def prog(msg, frac):
        if progress:
            progress(msg, min(max(frac, 0.0), 1.0))

    if mask is None and predict_prob is None:
        raise ValueError("Podaj maskę budynków albo model do ich wykrycia.")
    if min(terrain.shape) < 256:
        raise ValueError("Wybrany kadr jest zbyt mały (poniżej 128 m).")

    if mask is None:
        prog("Wykrywam budynki (U-Net)", 0.0)
        mask = predict_mask(terrain, predict_prob,
                            progress=lambda i, n: prog(f"Wykrywam budynki {i}/{n}", 0.20 * i / n))
        start, span = 0.20, 0.80
    else:
        start, span = 0.0, 1.0

    notes = []
    if ground is not None and ground.shape != terrain.shape:
        raise ValueError("Model gruntu ma inny rozmiar niż NMPT.")
    ground_src = "oficjalny NMT (geoportal)" if ground is not None else "wprost z NMPT"
    if ground is None and ground_mode == "estimate":
        prog("Szacuję grunt bez budynków i drzew", start)
        ground = am.estimate_ground(terrain, mask)
        ground_src = "oszacowany z NMPT"
    hyd = ground if ground is not None else terrain

    # kalibracja: powierzchnia rzeki w kadrze = stan normalny
    surface = am.estimate_river_surface(hyd)
    river_frac = float(((hyd > surface - 0.5) & (hyd < surface + am.RIVER_MARGIN_M + 0.1)).mean())
    if river_frac < 0.02:
        notes.append("W wybranym kadrze nie wykryto rzeki (mniej niż 2% powierzchni leży na poziomie "
                     "wody). Symulacja może nie mieć sensu. Wybierz kadr obejmujący rzekę.")
    if mask.mean() < 0.005:
        notes.append("Wykryto bardzo mało budynków - statystyki zalania będą mało wiarygodne.")

    events = build_events(max_rise_m, step_m)
    gauge_base = surface + am.RIVER_MARGIN_M - events[0]["gauge_cm"] / 100.0
    w_m, h_m = terrain.shape[1] * TARGET_RES, terrain.shape[0] * TARGET_RES
    if title is None:
        title = (f"Symulacja powodzi ({w_m:.0f}×{h_m:.0f} m)" if max(w_m, h_m) < 1000
                 else f"Symulacja powodzi ({w_m / 1000:.1f}×{h_m / 1000:.1f} km)")

    anim = am.make_animation(
        terrain, mask, out_path, title=title, ground=ground,
        gauge_base=gauge_base, fps=fps, dpi=None, events=events,
        calib="local", step_cm=10, bld_method="ring",
        progress=lambda msg, f: prog(msg, start + span * f),
        relative_base_cm=events[0]["gauge_cm"],
    )
    return {"surface": surface, "river_frac": river_frac, "notes": notes,
            "crop_px": terrain.shape, "buildings_pct": float(mask.mean() * 100),
            "n_events": len(events), "curve": (anim or {}).get("curve", []),
            "ground_src": ground_src}


def generate_gif(nmpt_path, out_path, cx_src, cy_src, crop_km=1.5,
                 mask_path=None, predict_prob=None,
                 max_rise_m=4.0, step_m=1.0, fps=4, progress=None, nmt_path=None):
    """Wersja dla wgranego pliku: wczytuje kadr (oraz maske i NMT, jesli podano), potem generate_from_terrain."""
    def prog(msg, frac):
        if progress:
            progress(msg, min(max(frac, 0.0), 1.0))

    info, errors, _ = inspect_tif(nmpt_path)
    if errors:
        raise ValueError(errors[0])
    if mask_path is None and predict_prob is None:
        raise ValueError("Podaj maskę budynków albo model do ich wykrycia.")

    prog("Wczytuję wybrany kadr", 0.0)
    window = make_window(info, cx_src, cy_src, crop_km * 1000)
    terrain, bounds = read_terrain_crop(nmpt_path, window, info)
    mask = None
    if mask_path:
        prog("Wczytuję maskę budynków", 0.04)
        mask = read_mask_crop(mask_path, bounds, terrain.shape)

    ground, uwagi = None, []
    if nmt_path:
        prog("Wczytuję NMT", 0.06)
        try:
            nmt_crop, cover = read_ground_crop(nmt_path, bounds, terrain.shape)
            ground = prepare_ground(terrain, nmt_crop, cover)
        except Exception as exc:
            uwagi.append(f"Nie użyto NMT: {exc}. Grunt oszacowano z NMPT.")

    wynik = generate_from_terrain(
        terrain, out_path, mask=mask, predict_prob=predict_prob, ground=ground,
        max_rise_m=max_rise_m, step_m=step_m, fps=fps,
        progress=lambda msg, f: prog(msg, 0.08 + 0.92 * f),
    )
    wynik["notes"] = uwagi + wynik["notes"]
    return wynik