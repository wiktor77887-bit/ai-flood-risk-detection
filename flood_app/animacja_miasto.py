"""
animacja_miasto.py  -  symulacja powodzi dla wielu miast (Poznan, Wroclaw, Torun)
──────────────────────────────────────────────────────────────────────────────
Uogolniony Animacja.py: ta sama symulacja, ale parametry miasta (pliki, rzeka,
stany wody, srodek kadru) sa w slowniku CITIES.

UZYCIE:
  python animacja_miasto.py --city wroclaw --grid 15
  python animacja_miasto.py --city torun   --grid 15
  python animacja_miasto.py --city poznan  --grid 15
  python animacja_miasto.py --all --grid 15            <- wszystkie miasta, ktore maja pliki
  python animacja_miasto.py --city wroclaw --river-lat 51.114 --river-lon 17.046
  python animacja_miasto.py --city torun --tif inny.tif --mask_tif inna_maska.tif

Domyslnie (bez --full / --center) kadr jest centrowany na rzece.

WYMAGA (dla kazdego miasta):  NMPT .tif  oraz  maska budynkow .tif
OPCJONALNIE:  pip install pyproj  (centrowanie wg wspolrzednych rzeki)
"""

import numpy as np
import argparse
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import (label as nd_label, maximum_filter, minimum_filter, uniform_filter,
                           binary_dilation, distance_transform_edt)
try:
    import rasterio
except ImportError:      # strona 'Wybierz miejsce' nie czyta plikow .tif
    rasterio = None
from PIL import Image

GAUGE_BASE_M = 49.46
RING_PX      = 3      # szerokosc pierscienia wokol budynku [px]
RING_FRAC    = 0.15   # budynek zalany, gdy >= tyle jego otoczenia jest pod woda
MIN_BLD_PX   = 30     # mniejsze obiekty nie licza sie jako budynki w statystykach
RIVER_MARGIN_M = 0.35   # tolerancja szumu LiDAR nad powierzchnia wody w korycie
PATCH        = 256

EVENTS_WARTA = [
    {"name": "Stan normalny",      "gauge_cm":  150, "year": None, "color": "#2196F3"},
    {"name": "Stan ostrzegawczy",  "gauge_cm":  400, "year": None, "color": "#FF9800"},
    {"name": "Stan alarmowy",      "gauge_cm":  450, "year": None, "color": "#FF5722"},
    {"name": "Powodz 2010",        "gauge_cm":  530, "year": 2010, "color": "#E91E63"},
    {"name": "Powodz 1997",        "gauge_cm":  580, "year": 1997, "color": "#9C27B0"},
    {"name": "Powodz 1924",        "gauge_cm":  700, "year": 1924, "color": "#673AB7"},
    {"name": "Powodz 1888",        "gauge_cm":  750, "year": 1888, "color": "#1A237E"},
    {"name": "Scenariusz X1",      "gauge_cm":  900, "year": None, "color": "#0D47A1"},
    {"name": "Scenariusz X2",      "gauge_cm": 1100, "year": None, "color": "#4A148C"},
    {"name": "Scenariusz MAX",     "gauge_cm": 1500, "year": None, "color": "#212121"},
]


# Stany wody w cm nad dnem koryta (przyblizenie wodowskazu - patrz main()).
# Wartosci ostrzegawczy/alarmowy: dane urzedowe (IMGW / urzedy miast).
EVENTS_ODRA_WROCLAW = [      # wodowskaz Odra - Trestno
    {"name": "Stan normalny",      "gauge_cm":  200, "year": None, "color": "#2196F3"},
    {"name": "Stan ostrzegawczy",  "gauge_cm":  380, "year": None, "color": "#FF9800"},
    {"name": "Stan alarmowy",      "gauge_cm":  450, "year": None, "color": "#FF5722"},
    {"name": "Powodz 1997",        "gauge_cm":  724, "year": 1997, "color": "#9C27B0"},
    {"name": "Scenariusz X1",      "gauge_cm":  850, "year": None, "color": "#0D47A1"},
    {"name": "Scenariusz X2",      "gauge_cm": 1000, "year": None, "color": "#4A148C"},
    {"name": "Scenariusz MAX",     "gauge_cm": 1200, "year": None, "color": "#212121"},
]

EVENTS_WISLA_TORUN = [       # wodowskaz Wisla - Torun
    {"name": "Stan normalny",      "gauge_cm":  250, "year": None, "color": "#2196F3"},
    {"name": "Stan ostrzegawczy",  "gauge_cm":  530, "year": None, "color": "#FF9800"},
    {"name": "Stan alarmowy",      "gauge_cm":  650, "year": None, "color": "#FF5722"},
    {"name": "Scenariusz X1",      "gauge_cm":  750, "year": None, "color": "#0D47A1"},
    {"name": "Scenariusz X2",      "gauge_cm":  850, "year": None, "color": "#4A148C"},
    {"name": "Scenariusz MAX",     "gauge_cm": 1000, "year": None, "color": "#212121"},
]


EVENTS_SCENARIUSZE = [       # brak zweryfikowanych stanow alarmowych - tylko scenariusze
    {"name": "Stan normalny",   "gauge_cm": 200, "year": None, "color": "#2196F3"},
    {"name": "Scenariusz +1 m", "gauge_cm": 300, "year": None, "color": "#FF9800"},
    {"name": "Scenariusz +2 m", "gauge_cm": 400, "year": None, "color": "#FF5722"},
    {"name": "Scenariusz +3 m", "gauge_cm": 500, "year": None, "color": "#9C27B0"},
    {"name": "Scenariusz +4 m", "gauge_cm": 600, "year": None, "color": "#0D47A1"},
    {"name": "Scenariusz +6 m", "gauge_cm": 800, "year": None, "color": "#212121"},
]

# river_lonlat = (lon, lat) punktu na rzece w srodku kadru; None => auto-wyszukiwanie.
# Nazwy plikow dopasuj do swoich (lub nadpisz przez --tif / --mask_tif).
CITIES = {
    "poznan": dict(
        label="Poznań", river_gen="Warty",
        tif="NMPT_POZNAN_CALOSC.tif", mask="idealna_maska_budynkow.tif",
        out="powodz_warta_poznan.gif",
        river_lonlat=(16.9480, 52.4054), events=EVENTS_WARTA,
        calib="city", step_cm=None),   # zachowuje dotychczasowe zachowanie
    "wroclaw": dict(
        label="Wrocław", river_gen="Odry",
        tif="NMPT_WROCLAW_CALOSC.tif", mask="WROCLAW_maska_budynkow.tif",
        out="powodz_odra_wroclaw.gif",
        river_lonlat=(17.0450, 51.1140), events=EVENTS_ODRA_WROCLAW,   # okolice Ostrowa Tumskiego
        calib="local", step_cm=10),
    "bydgoszcz": dict(
        label="Bydgoszcz", river_gen="Brdy",
        tif="NMPT_BYDGOSZCZ_CALOSC.tif", mask="idealna_maska_budynkow_bydgoszcz.tif",
        out="powodz_brda_bydgoszcz.gif",
        river_lonlat=None, events=EVENTS_SCENARIUSZE,
        calib="local", step_cm=10),
    "torun": dict(
        label="Toruń", river_gen="Wisły",
        tif="NMPT_TORUN_CALOSC.tif", mask="TORUN_maska_budynkow.tif",
        out="powodz_wisla_torun.gif",
        river_lonlat=None, events=EVENTS_WISLA_TORUN,
        calib="local", step_cm=10),
}


def gauge_to_masl(gauge_cm, base=None):
    return (base if base is not None else GAUGE_BASE_M) + gauge_cm / 100.0


def nearest_event_label(water_level, levels, names, colors):
    closest = min(range(len(levels)), key=lambda i: abs(levels[i] - water_level))
    return names[closest], colors[closest]


# ==========================================
# POWÓDŹ — FLOOD FILL OD KORYTA RZEKI
# ==========================================
def build_river_seed(terrain, river_level, lake_level):
    """
    Seed powodzi = koryto rzeki dotykające krawędzi mapy + duże zbiorniki wodne.
    Dzięki temu woda zalewa tylko obszary połączone z rzeką,
    a nie każdą nieckę czy bajorko na mapie.
    river_level / lake_level to poziomy bezwzględne [m n.p.m.].
    """
    # 1. Koryto rzeki — tylko elementy dotykające krawędzi
    below_river = terrain < river_level
    labeled_r, _ = nd_label(below_river)

    border_labels = set()
    for edge in [labeled_r[0, :], labeled_r[-1, :],
                 labeled_r[:, 0], labeled_r[:, -1]]:
        border_labels.update(edge.ravel())
    border_labels.discard(0)

    seed = np.zeros(terrain.shape, dtype=bool)
    for lbl in border_labels:
        seed |= (labeled_r == lbl)

    # 2. Duże zbiorniki wodne (starorzecza, jeziora itp.)
    MIN_LAKE_PX = 50000   # ~0.3 km² przy 0.5m/px — wyklucza bajorka i parki
    below_lake  = terrain < lake_level
    labeled_l, n_l = nd_label(below_lake)

    lake_count = 0
    for lbl in range(1, n_l + 1):
        comp = labeled_l == lbl
        sz   = comp.sum()
        if sz >= MIN_LAKE_PX and not seed[comp].any():
            seed |= comp
            lake_count += 1
            print(f"    Zbiornik #{lake_count}: {sz} px @ <{lake_level:.2f} m")

    print(f"  Seed: koryto {seed.mean()*100:.2f}% | zbiorniki: {lake_count}")
    return seed


def precompute_buildings(bld_mask, ring_px=RING_PX, min_px=MIN_BLD_PX):
    """
    Obrysy budynkow i ich otoczenie (pierscien ring_px od scian).
    NMPT zawiera DACHY, wiec woda nigdy nie siega 'pikseli budynku' (dach jest
    wyzszy niz woda na ulicy). Budynek uznajemy za zalany, gdy woda stoi
    przy jego scianach - czyli gdy zalana jest czesc jego otoczenia.
    """
    lab, nb = nd_label(bld_mask)
    area = np.bincount(lab.ravel(), minlength=nb + 1)
    near = maximum_filter(lab, size=2 * ring_px + 1)          # najblizszy budynek w oknie
    ring_lab = np.where(lab == 0, near, 0).astype(lab.dtype)  # tylko piksele POZA budynkami
    ring_total = np.bincount(ring_lab.ravel(), minlength=nb + 1)
    valid = area >= min_px
    valid[0] = False
    return {"lab": lab, "ring_lab": ring_lab, "ring_total": ring_total,
            "valid": valid, "n": nb}


def estimate_ground(nmpt, bld_mask, bld_margin_px=2, rough_thr=0.35, veg_min_h=1.5):
    """
    Przyblizony model GRUNTU z NMPT: usuwa budynki (maska) oraz drzewa i krzewy, a puste miejsca
    wypelnia wartoscia z najblizszego piksela gruntu.
    NMPT zawiera dachy i korony drzew, wiec liczenie wody wprost na nim daje niespojne wysokosci
    (woda nie 'widzi' gruntu pod dachem ani pod korona). Drzewa rozpoznajemy jako obiekty
    wyraznie wyzsze od otoczenia i chropowate; wały i nasypy sa gladkie, wiec zostaja.
    Przyblizenie: ostatecznie lepszy jest oficjalny NMT z geoportalu.
    """
    t = nmpt.astype(np.float32)
    rough = uniform_filter(np.abs(t - uniform_filter(t, 5)), 7)
    veg = (rough > rough_thr) & ((t - minimum_filter(t, size=31)) > veg_min_h)
    veg = binary_dilation(veg, iterations=2)
    bld = binary_dilation(np.asarray(bld_mask) > 0.5, iterations=bld_margin_px)
    remove = veg | bld
    if not remove.any():
        return t
    idx = distance_transform_edt(remove, return_distances=False, return_indices=True)
    return t[tuple(idx)]


def flooded_buildings(bi, flooded, depth=None, frac=RING_FRAC):
    """
    Wektor bool (n+1,): czy budynek o danej etykiecie jest zalany.
    Z `depth` (glebokosc wody [m] na kazdym pikselu) zwraca tez srednia glebokosc przy scianach.
    """
    idx = bi["ring_lab"][flooded]
    cnt = np.bincount(idx, minlength=bi["n"] + 1)
    need = np.maximum(3, frac * bi["ring_total"])
    fb = bi["valid"] & (cnt >= need)
    if depth is None:
        return fb
    sm = np.bincount(idx, weights=depth[flooded], minlength=bi["n"] + 1)
    mean_d = np.divide(sm, cnt, out=np.zeros_like(sm), where=cnt > 0)
    return fb, mean_d


def compute_flood_mask(terrain, water_level, river_seed):
    """Zalewa tylko piksele połączone z korytem rzeki."""
    below      = terrain < water_level
    candidates = below | river_seed
    labeled, n = nd_label(candidates)

    river_labels = np.unique(labeled[river_seed])
    river_labels = river_labels[river_labels > 0]
    if river_labels.size == 0:
        return np.zeros(terrain.shape, dtype=bool)

    lut = np.zeros(n + 1, dtype=bool)          # O(N) zamiast petli po etykietach
    lut[river_labels] = True
    return lut[labeled] & (below | river_seed)


class PackedFloods(dict):
    """Maski powodzi spakowane do bitow (8x mniej RAM-u); odpakowywane przy odczycie."""
    def __init__(self, shape):
        super().__init__()
        self.shape = shape
        self.count = int(np.prod(shape))

    def add(self, key, mask):
        dict.__setitem__(self, key, np.packbits(mask.ravel()))

    def __getitem__(self, key):
        bits = dict.__getitem__(self, key)
        return np.unpackbits(bits, count=self.count).astype(bool).reshape(self.shape)

    def items(self):
        for k in list(self.keys()):
            yield k, self[k]


def precompute_all_floods(terrain, water_levels_unique, river_seed, progress=None):
    flood_cache = PackedFloods(terrain.shape)
    n = len(water_levels_unique)
    for i, wl in enumerate(sorted(water_levels_unique)):
        flood_cache.add(wl, compute_flood_mask(terrain, wl, river_seed))
        if progress:
            progress(i + 1, n)
        if (i + 1) % 5 == 0 or (i + 1) == n:
            print(f"    Maski powodzi: {i+1}/{n}", end='\r')
    print()
    return flood_cache


# ==========================================
# ŁADOWANIE DANYCH Z .TIF
# ==========================================
def load_tif(path, is_mask=False, downsample=1):
    print(f"  Wczytuję: {path}")
    with rasterio.open(path) as src:
        data   = src.read(1).astype(np.float32)
        nodata = src.nodata
        print(f"    Rozmiar: {src.width}x{src.height} px")

    if is_mask:
        data[data == 255] = 0
        if nodata is not None:
            data[data == nodata] = 0
        data = (data > 0).astype(np.float32)
        print(f"    Budynki: {data.mean()*100:.1f}% pikseli")
    else:
        if nodata is not None:
            data[data == nodata] = np.nan
        data[data < -100] = np.nan
        data = np.nan_to_num(data, nan=float(np.nanmean(data)))
        print(f"    Zakres:  {data.min():.1f} – {data.max():.1f} m n.p.m.")

    if downsample > 1:
        if is_mask:
            h = (data.shape[0] // downsample) * downsample
            w = (data.shape[1] // downsample) * downsample
            data = data[:h, :w].reshape(
                h // downsample, downsample,
                w // downsample, downsample
            ).max(axis=(1, 3))
        else:
            data = data[::downsample, ::downsample]

    return data


def align_arrays(terrain, mask):
    h = min(terrain.shape[0], mask.shape[0])
    w = min(terrain.shape[1], mask.shape[1])
    print(f"  Wyrównany rozmiar: {w}x{h} px")
    return terrain[:h, :w], mask[:h, :w]


def estimate_river_surface(terrain, bin_m=0.05):
    """
    Lokalna powierzchnia wody w korycie widoczna w NMPT (LiDAR mierzy powierzchnię
    wody, nie dno). Woda to duzy plaski obszar, wiec daje ostry pik w histogramie
    wysokosci. Bierzemy NAJNIZSZY wyrazny pik w dolnej cwiartce wysokosci kadru.
    Lepsze niz percentyl calej mapy: rzeka ma spadek, wiec dolny bieg miasta lezy
    nizej niz kadr w centrum.
    """
    sub = terrain[::2, ::2]
    lo, hi = np.percentile(sub, [0.5, 25])
    vals = sub[(sub >= lo) & (sub <= hi)]
    if vals.size < 100 or hi - lo < 3 * bin_m:
        return float(np.percentile(sub, 1))
    nb = max(10, int((hi - lo) / bin_m))
    hist, edges = np.histogram(vals, bins=nb)
    sm = np.convolve(hist, np.ones(3), mode="same")
    cand = np.where(sm >= 0.5 * sm.max())[0]
    start = int(cand[0])
    end = start
    while end + 1 < len(sm) and sm[end + 1] >= 0.5 * sm.max():
        end += 1
    k = start + int(np.argmax(sm[start:end + 1]))
    return float((edges[k] + edges[k + 1]) / 2)


def find_river_window_center(terrain, mask, crop_px, step=64):
    """
    Szuka kadru crop_px x crop_px, w ktorym jest i koryto rzeki (najnizszy teren),
    i duzo budynkow. Liczy na zgrubnej siatce (blok = step px) przez sumy
    skumulowane, wiec dziala szybko nawet dla ogromnych map.
    """
    h, w = terrain.shape
    hb, wb = h // step, w // step
    if hb < 2 or wb < 2:
        return w // 2, h // 2
    t = terrain[:hb * step, :wb * step].reshape(hb, step, wb, step)
    m = mask[:hb * step, :wb * step].reshape(hb, step, wb, step)
    river_level = float(np.percentile(terrain[::8, ::8], 1)) + 1.5
    river = (t < river_level).mean(axis=(1, 3))
    bld   = (m > 0.5).mean(axis=(1, 3))

    def integral(a):
        return np.pad(a.cumsum(0).cumsum(1), ((1, 0), (1, 0)))

    k = max(1, min(crop_px // step, hb, wb))

    def window_sums(a):
        I = integral(a)
        return I[k:, k:] - I[:-k, k:] - I[k:, :-k] + I[:-k, :-k]

    r, b  = window_sums(river), window_sums(bld)
    score = np.minimum(r, 0.15 * k * k) * b        # rzeka (do 15% kadru) x zabudowa
    iy, ix = np.unravel_index(int(np.argmax(score)), score.shape)
    cy = int((iy + k / 2) * step)
    cx = int((ix + k / 2) * step)
    cx = max(crop_px // 2, min(w - crop_px // 2, cx))
    cy = max(crop_px // 2, min(h - crop_px // 2, cy))
    return cx, cy


def get_spatial_crop(terrain, mask, center_patch=None, grid_size=15,
                     river_center=False, tif_path=None,
                     river_lonlat=(16.9480, 52.4054)):
    crop_px = grid_size * PATCH
    h, w    = terrain.shape
    cx, cy  = w // 2, h // 2   # domyślnie środek

    # --- Opcja 1: centrum podane jako numer kafelka ---
    if center_patch is not None:
        cols = math.ceil(w / PATCH)
        cr   = center_patch // cols
        cc   = center_patch  % cols
        cy   = cr * PATCH + PATCH // 2
        cx   = cc * PATCH + PATCH // 2
        print(f"  Centrum: kafelek {center_patch} → piksel ({cx}, {cy})")

    # --- Opcja 2: --river → centruj na Warcie przez pyproj ---
    elif river_center and tif_path is not None:
        placed = False
        try:
            if river_lonlat is None:
                raise LookupError
            from pyproj import Transformer
            with rasterio.open(tif_path) as src:
                trf = src.transform
                crs = src.crs
            tr  = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            # Punkt między Mostem Rocha a Jez. Maltańskim
            rx, ry = tr.transform(*river_lonlat)
            cx = int((rx - trf.c) / trf.a)
            cy = int((ry - trf.f) / trf.e)
            cx = max(crop_px // 2, min(w - crop_px // 2, cx))
            cy = max(crop_px // 2, min(h - crop_px // 2, cy))
            print(f"  Centrum rzeki (pyproj) → piksel ({cx}, {cy})")
            placed = True
        except LookupError:
            print("  Brak współrzędnych rzeki — szukam kadru z rzeką automatycznie")
        except ImportError:
            print("  pyproj niedostępny — szukam najniższego obszaru na krawędzi")
        except Exception as e:
            print(f"  pyproj błąd: {e} — szukam automatycznie")

        # --- Fallback: kadr z największą ilością koryta rzeki i zabudowy ---
        if not placed:
            cx, cy = find_river_window_center(terrain, mask, crop_px)
            print(f"  Auto centrum (rzeka + zabudowa) → piksel ({cx}, {cy})")

    # --- Opcja 3: brak parametrów → największe skupisko budynków ---
    else:
        best_sum = -1.0
        margin   = PATCH
        for yy in range(margin, max(margin + 1, h - crop_px - margin), PATCH):
            for xx in range(margin, max(margin + 1, w - crop_px - margin), PATCH):
                s = float(mask[yy:yy + crop_px, xx:xx + crop_px].sum())
                if s > best_sum:
                    best_sum = s
                    cy = yy + crop_px // 2
                    cx = xx + crop_px // 2
        print(f"  Auto centrum (budynki) → piksel ({cx}, {cy})")

    y0 = max(0, cy - crop_px // 2)
    x0 = max(0, cx - crop_px // 2)
    y1 = min(h, y0 + crop_px)
    x1 = min(w, x0 + crop_px)

    km_x = (x1 - x0) * 0.5 / 1000
    km_y = (y1 - y0) * 0.5 / 1000
    print(f"  Kadr: {x1-x0}x{y1-y0} px ≈ {km_x:.1f}x{km_y:.1f} km")
    return terrain[y0:y1, x0:x1], mask[y0:y1, x0:x1]


# ==========================================
# ANIMACJA
# ==========================================
def _render_frame(fig, update, fi):
    update(fi)
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


def save_gif_fixed_palette(fig, update, n_frames, path, fps, sample_idx, progress=None):
    """
    GIF ze STALA paleta kolorow dla wszystkich klatek.
    Domyslny zapis (PillowWriter) kwantyzuje kazda klatke osobno, przez co kolory
    elementow statycznych (np. legenda) zmieniaja sie z klatki na klatke.
    Tu paleta powstaje raz, z probki klatek, a klatki sa do niej mapowane bez ditheringu.
    """
    dither_none = getattr(Image, "Dither", Image).NONE
    mediancut   = getattr(Image, "Quantize", Image).MEDIANCUT

    cache = {i: _render_frame(fig, update, i) for i in sample_idx}
    w, h = next(iter(cache.values())).size
    montage = Image.new("RGB", (w, h * len(cache)))
    for k, i in enumerate(sorted(cache)):
        montage.paste(cache[i], (0, k * h))
    pal = montage.quantize(colors=256, method=mediancut, dither=dither_none)

    frames = []
    for i in range(n_frames):
        im = cache.get(i) or _render_frame(fig, update, i)
        frames.append(im.quantize(palette=pal, dither=dither_none))
        if progress:
            progress(i + 1, n_frames)
        if (i + 1) % 10 == 0 or i + 1 == n_frames:
            print(f"    Klatki: {i + 1}/{n_frames}", end="\r")
    print()
    frames[0].save(path, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)


def make_animation(terrain, mask, output_path, title,
                   gauge_base=None, fps=4, dpi=None, events=None,
                   calib="city", step_cm=None, bld_method="ring",
                   progress=None, relative_base_cm=None, ground=None):

    def _p(msg, frac):
        if progress:
            progress(msg, min(max(frac, 0.0), 1.0))

    events = events or EVENTS_WARTA
    gb   = gauge_base if gauge_base is not None else GAUGE_BASE_M
    px_w = terrain.shape[1]

    # Auto DPI — im większy kadr tym mniejsze DPI żeby gif nie ważył gigabajtów
    if dpi is None:
        if   px_w >= 8000: dpi = 55
        elif px_w >= 5000: dpi = 70
        elif px_w >= 3000: dpi = 90
        elif px_w >= 2000: dpi = 110
        else:              dpi = 130
        print(f"  Auto DPI: {dpi}  (kadr {px_w}x{terrain.shape[0]} px)")

    levels = [gauge_to_masl(e["gauge_cm"], gb) for e in events]
    names  = [
        e["name"] + ("" if e.get("hide_cm") else f' ({e["gauge_cm"]} cm)')
        + (f'  [{e["year"]}]' if e["year"] else "")
        for e in events
    ]
    colors = [e["color"] for e in events]

    # Sekwencja klatek: start w stanie normalnym (woda w korycie), potem wzrost
    # ze stalym krokiem (step_cm) albo 10 klatek na zdarzenie, z zatrzymaniem na poziomach.
    HOLD = 6
    water_seq = [round(levels[0], 4)] * HOLD
    for i in range(len(levels) - 1):
        d_cm  = (levels[i + 1] - levels[i]) * 100
        n_sub = 10 if step_cm is None else max(2, int(math.ceil(d_cm / step_cm)))
        for t in np.linspace(0, 1, n_sub, endpoint=False):
            water_seq.append(round(levels[i] + t * (levels[i + 1] - levels[i]), 4))
        water_seq += [round(levels[i + 1], 4)] * HOLD
    water_seq += [round(levels[-1], 4)] * 12

    print(f"  Klatek: {len(water_seq)}")

    # Progi koryta rzeki (bezwzgledne)
    if calib == "local":
        river_lvl, lake_lvl, always_lvl = levels[0], levels[0] + 1.0, levels[0]
    else:
        river_lvl, lake_lvl, always_lvl = gb + 0.30, gb + 1.5, gb + 1.5

    print("  Wyznaczam koryto rzeki i zbiorniki...")
    _p("Wyznaczam koryto rzeki", 0.02)
    hyd = ground if ground is not None else terrain      # woda liczona na gruncie, nie na dachach
    river_seed = build_river_seed(hyd, river_lvl, lake_lvl)

    unique_levels = sorted(set(water_seq))
    print(f"  Pre-obliczam {len(unique_levels)} masek powodzi...")
    flood_cache = precompute_all_floods(
        hyd, unique_levels, river_seed,
        progress=lambda i, n: _p(f"Maski powodzi {i}/{n}", 0.05 + 0.30 * i / n))
    print("  Gotowe. Składam klatki...")

    bld_mask  = mask > 0.5
    bld_total = int(bld_mask.sum())
    bi = None
    if bld_method == "ring":
        print("  Wyznaczam obrysy budynków i ich otoczenie...")
        _p("Obrysy budynków", 0.37)
        bi = precompute_buildings(bld_mask)
        n_bld = int(bi["valid"].sum())
        print(f"  Budynków (>= {MIN_BLD_PX} px): {n_bld}")

    # Tylko lekkie metadane klatek - maski liczone w update() (oszczednosc RAM)
    frame_data = [(wl,) + nearest_event_label(wl, levels, names, colors)
                  for wl in water_seq]

    terrain_cmap = LinearSegmentedColormap.from_list(
        "urban", [
            '#1a0a00', '#5c3a1e', '#9b7540', '#c8aa72', '#d4c49a', '#e8e0c8',
        ]
    )

    aspect  = terrain.shape[1] / terrain.shape[0]
    fig_w   = min(9 * aspect, 20)
    fig, ax = plt.subplots(figsize=(fig_w, 9))
    fig.set_dpi(dpi)
    fig.patch.set_facecolor('#0D1B2A')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)

    t_min = float(np.percentile(terrain, 1))
    t_max = float(np.percentile(terrain, 99))

    ax.axis('off')
    ax.set_facecolor('#0D1B2A')

    im_terrain = ax.imshow(terrain, cmap=terrain_cmap, vmin=t_min, vmax=t_max,
                            interpolation='bilinear', aspect='equal')
    river_always = hyd < always_lvl
    river_rgba          = np.zeros((*terrain.shape, 4), dtype=np.float32)
    river_rgba[river_always, 0] = 0.04
    river_rgba[river_always, 1] = 0.38
    river_rgba[river_always, 2] = 0.95
    river_rgba[river_always, 3] = 0.85
    im_river_base = ax.imshow(river_rgba, interpolation='bilinear', aspect='equal')

    water_rgba         = np.zeros((*terrain.shape, 4), dtype=np.float32)
    water_rgba[..., 0] = 0.04
    water_rgba[..., 1] = 0.38
    water_rgba[..., 2] = 0.95
    im_water = ax.imshow(water_rgba, interpolation='bilinear', aspect='equal')

    # Klasy budynkow: 0 ocalaly, 1 zalany plytko (<0.5 m), 2 sredni (0.5-1.5 m), 3 gleboki (>1.5 m)
    BLD_LUT = np.array([[1.0, 0.88, 0.0, 1.0],
                        [1.0, 0.60, 0.0, 1.0],
                        [1.0, 0.10, 0.0, 1.0],
                        [0.55, 0.0, 0.0, 1.0]], dtype=np.float32)
    bld_rgba = np.zeros((*terrain.shape, 4), dtype=np.float32)
    im_bld   = ax.imshow(bld_rgba, interpolation='nearest', aspect='equal')

    fs = 12
    txt_water = ax.text(
        0.015, 0.975, "", transform=ax.transAxes,
        color='white', fontsize=fs, fontweight='bold', va='top',
        fontfamily='monospace',
        bbox=dict(facecolor='#0A1628', alpha=0.88,
                  edgecolor='white', linewidth=2, boxstyle='round,pad=0.5'))

    # Etykieta zdarzenia: tlo w kolorze zdarzenia, bialy tekst i biala ramka
    # (kolorowy tekst na ciemnym tle byl nieczytelny dla ciemnych kolorow)
    txt_event = ax.text(
        0.015, 0.865, "", transform=ax.transAxes,
        color='white', fontsize=fs - 1, fontweight='bold', va='top',
        bbox=dict(facecolor='#0A1628', alpha=0.95,
                  edgecolor='white', linewidth=2, boxstyle='round,pad=0.4'))

    txt_stats = ax.text(
        0.985, 0.975, "", transform=ax.transAxes,
        color='white', fontsize=fs - 1, va='top', ha='right',
        fontfamily='monospace',
        bbox=dict(facecolor='#0A1628', alpha=0.88,
                  edgecolor='#37474F', linewidth=1, boxstyle='round,pad=0.5'))

    if bld_method == "ring":
        legend_handles = [
            mpatches.Patch(facecolor='#0A52D4', alpha=0.85, label='Woda'),
            mpatches.Patch(facecolor='#FFE000', label='Budynki - ocalałe'),
            mpatches.Patch(facecolor='#FF9900', label='Zalane do 0,5 m'),
            mpatches.Patch(facecolor='#FF1A00', label='Zalane 0,5 - 1,5 m'),
            mpatches.Patch(facecolor='#8C0000', label='Zalane powyżej 1,5 m'),
        ]
    else:
        legend_handles = [
            mpatches.Patch(facecolor='#0A52D4', alpha=0.85, label='Woda'),
            mpatches.Patch(facecolor='#FFE000', label='Budynki - ocalałe'),
            mpatches.Patch(facecolor='#FF1414', label='Budynki - zalane'),
        ]
    ax.legend(
        handles=legend_handles,
        loc='lower right', fontsize=fs - 2,
        facecolor='#0A1628', edgecolor='#37474F',
        labelcolor='white', framealpha=0.9
    )
    ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=8)

    def update(fi):
        wl, lbl, evt_color = frame_data[fi]
        flooded  = flood_cache[wl]
        gauge_cm = (wl - gb) * 100
        water_depth = np.where(flooded, np.maximum(wl - hyd, 0.0), 0.0).astype(np.float32)
        n_deep = 0
        if bld_method == "ring":
            fb, mean_d = flooded_buildings(bi, flooded, water_depth)
            cls = np.zeros(bi["n"] + 1, dtype=np.uint8)
            cls[fb & (mean_d < 0.5)] = 1
            cls[fb & (mean_d >= 0.5) & (mean_d < 1.5)] = 2
            cls[fb & (mean_d >= 1.5)] = 3
            pix_cls = cls[bi["lab"]]
            n_fl   = int(fb.sum())
            n_deep = int((cls == 3).sum())
        else:   # stara metoda: piksele dachu ponizej wody
            pix_cls = np.where(bld_mask & flooded, 2, 0).astype(np.uint8)
            bld_flooded = int((pix_cls == 2).sum())

        # Woda: alpha zalezy od glebokosci
        max_d = max(float(water_depth.max()), 0.01)
        water_rgba[..., 3] = (
            np.clip(water_depth / max_d * 0.37 + 0.55, 0.0, 0.92)
            * flooded.astype(np.float32)
        )
        im_water.set_data(water_rgba)

        bld_rgba[...] = BLD_LUT[pix_cls]
        bld_rgba[~bld_mask] = 0.0
        im_bld.set_data(bld_rgba)

        if relative_base_cm is None:
            stan = f"Stan wody:    {gauge_cm:.0f} cm"
        else:
            rise = round((gauge_cm - relative_base_cm) / 100, 1) + 0.0   # +0.0 zamiast -0.0
            stan = f"Wzrost:       {rise:+.1f} m"
        txt_water.set_text(f"Poziom wody:  {wl:.2f} m n.p.m.\n{stan}")

        txt_event.set_text(lbl)
        txt_event.get_bbox_patch().set_facecolor(evt_color)

        if bld_method == "ring":
            txt_stats.set_text(
                f"Zalany teren:    {flooded.sum()/terrain.size*100:5.1f}%\n"
                f"Zalane budynki: {n_fl/max(n_bld,1)*100:5.1f}%\n"
                f"  ({n_fl:,} / {n_bld:,} budynków)\n"
                f"  glebiej niz 1,5 m: {n_deep:,}"
            )
        else:
            txt_stats.set_text(
                f"Zalany teren:    {flooded.sum()/terrain.size*100:5.1f}%\n"
                f"Zalane budynki: {bld_flooded/max(bld_total,1)*100:5.1f}%\n"
                f"  ({bld_flooded:,} / {bld_total:,} px)"
            )

    # Klatki probkowe do wspolnej palety: rownomiernie + klatka z najwieksza liczba
    # zalanych budynkow (zeby czerwien byla w palecie) + ostatnia
    n_frames = len(frame_data)
    fld_by_level, curve = {}, []
    for wl, fm in flood_cache.items():
        nb = (int(flooded_buildings(bi, fm).sum()) if bld_method == "ring"
              else int((bld_mask & fm).sum()))
        fld_by_level[wl] = nb
        denom = max(n_bld if bld_method == "ring" else bld_total, 1)
        curve.append({"rise_m": float(round(wl - levels[0], 3)) + 0.0, "area_pct": float(fm.mean() * 100),
                      "bld_pct": 100.0 * nb / denom})
    curve.sort(key=lambda r: r["rise_m"])
    counts = [fld_by_level[fd[0]] for fd in frame_data]
    sample_idx = sorted(set(
        np.linspace(0, n_frames - 1, 14).astype(int).tolist()
        + [int(np.argmax(counts)), n_frames - 1]
    ))

    print(f"Zapisywanie {output_path} ...")
    save_gif_fixed_palette(
        fig, update, n_frames, output_path, fps, sample_idx,
        progress=lambda i, n: _p(f"Klatki {i}/{n}", 0.40 + 0.58 * i / n))
    plt.close(fig)

    size_mb = Path(output_path).stat().st_size / 1024**2
    print(f"✅ Zapisano: {output_path}  ({size_mb:.1f} MB)")
    _p("Gotowe", 1.0)
    return {"curve": curve}


# ==========================================
# MAIN
# ==========================================
def run_city(key, args):
    cfg  = CITIES[key]
    tif  = args.tif      or cfg["tif"]
    mtif = args.mask_tif or cfg["mask"]
    out  = args.out      or cfg["out"]

    for f in (tif, mtif):
        if not Path(f).exists():
            print(f"❌ {cfg['label']}: nie znaleziono {f} — pomijam")
            return False

    river_lonlat = cfg["river_lonlat"]
    if args.river_lat is not None and args.river_lon is not None:
        river_lonlat = (args.river_lon, args.river_lat)

    ds      = args.downsample if args.full else 1
    calib   = args.calib or cfg.get("calib", "city")
    step_cm = args.step_cm if args.step_cm is not None else cfg.get("step_cm")
    events  = cfg["events"]

    print("=" * 58)
    print(f"  Symulacja powodzi {cfg['river_gen']} — {cfg['label']}")
    print("=" * 58)

    print("\nWczytuję teren...")
    terrain = load_tif(tif,  is_mask=False, downsample=ds)
    print("Wczytuję maskę budynków...")
    mask    = load_tif(mtif, is_mask=True,  downsample=ds)
    print("Wyrównuję rozmiary...")
    terrain, mask = align_arrays(terrain, mask)

    city_p1 = float(np.percentile(terrain, 1))     # dno koryta wg calej mapy

    if args.full:
        title = f"Symulacja powodzi {cfg['river_gen']} — {cfg['label']} (pełna mapa, 1/{ds})"
    else:
        use_river = args.center is None
        print(f"Tryb: przestrzenny {args.grid}×{args.grid} kafelków "
              f"({'rzeka' if use_river else '--center'})")
        terrain, mask = get_spatial_crop(
            terrain, mask,
            center_patch=args.center,
            grid_size=args.grid,
            river_center=use_river,
            tif_path=tif,
            river_lonlat=river_lonlat,
        )
        km = args.grid * PATCH * 0.5 / 1000
        title = (f"Symulacja powodzi {cfg['river_gen']} — {cfg['label']}  "
                 f"({args.grid}×{args.grid} kafelków, ~{km:.0f}×{km:.0f} km)")

    ground = None
    if args.ground == "estimate":
        print("Szacuję model gruntu (bez budynków i drzew)...")
        ground = estimate_ground(terrain, mask)
    hyd = ground if ground is not None else terrain

    # Kalibracja poziomu wody (poziom = baza + cm/100).
    #  local: powierzchnia rzeki w kadrze = "stan normalny" (woda od razu w korycie)
    #  city : 1. percentyl calej mapy (dawne zachowanie)
    if calib == "local":
        surface    = estimate_river_surface(hyd)
        gauge_base = surface + RIVER_MARGIN_M - events[0]["gauge_cm"] / 100.0
        print(f"\n  [Kalibracja lokalna] powierzchnia rzeki w kadrze ~ {surface:.2f} m n.p.m.")
        print(f"  (dno wg całej mapy: {city_p1:.2f} m — różnica {surface - city_p1:+.2f} m)")
    else:
        gauge_base = city_p1
        print(f"\n  [Kalibracja] dno koryta (cała mapa) ~ {city_p1:.2f} m n.p.m.")
    print(f"  {'Zdarzenie':25s} {'cm':>5}   {'m n.p.m.':>10}   {'zalanie':>7}")
    print("  " + "─" * 55)
    for e in events:
        lvl = gauge_base + e["gauge_cm"] / 100.0
        pct = float((hyd < lvl).mean()) * 100
        print(f"  {e['name']:25s} {e['gauge_cm']:5d}   {lvl:10.2f}   {pct:6.1f}%")
    print()

    make_animation(
        terrain, mask, out,
        title=title, gauge_base=gauge_base,
        fps=args.fps, dpi=args.dpi, events=events,
        calib=calib, step_cm=step_cm, bld_method=args.bld_method, ground=ground,
    )
    print(f"\n🎉 Gotowe! Otwórz: {out}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Animacja powodzi — wiele miast",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--city", choices=sorted(CITIES), default=None)
    parser.add_argument("--all", action="store_true",
                        help="Wszystkie miasta z CITIES (pomija te bez plików)")
    parser.add_argument("--tif",        default=None, help="Nadpisz plik NMPT")
    parser.add_argument("--mask_tif",   default=None, help="Nadpisz plik maski budynków")
    parser.add_argument("--out",        default=None, help="Nadpisz nazwę GIF-a")
    parser.add_argument("--full",       action="store_true",
                        help="Cała mapa (użyj z --downsample)")
    parser.add_argument("--center",     type=int, default=None,
                        help="Numer kafelka jako centrum kadru (zamiast rzeki)")
    parser.add_argument("--river-lat",  type=float, default=None)
    parser.add_argument("--river-lon",  type=float, default=None)
    parser.add_argument("--grid",       type=int, default=15,
                        help="Rozmiar siatki N×N kafelków (domyślnie 15)")
    parser.add_argument("--downsample", type=int, default=1,
                        help="Próbkowanie co N pikseli (dla --full)")
    parser.add_argument("--calib", choices=["local", "city"], default=None,
                        help="local = woda od powierzchni rzeki w kadrze; city = dno wg całej mapy")
    parser.add_argument("--ground", choices=["estimate", "none"], default="estimate",
                        help="estimate = woda liczona na przyblizonym gruncie (bez budynkow i drzew); "
                             "none = wprost na NMPT (stara metoda)")
    parser.add_argument("--bld-method", choices=["ring", "roof"], default="ring",
                        help="ring = budynek zalany, gdy woda stoi przy scianach (domyślnie); "
                             "roof = stara metoda (piksele dachu poniżej wody)")
    parser.add_argument("--step-cm", type=int, default=None,
                        help="Krok poziomu wody w cm (domyślnie wg miasta; mniejszy = płynniej)")
    parser.add_argument("--fps",        type=int, default=4)
    parser.add_argument("--dpi",        type=int, default=None)
    args = parser.parse_args()

    if args.all:
        if args.tif or args.mask_tif or args.out:
            parser.error("--tif/--mask_tif/--out działają tylko z --city")
        wyniki = {k: run_city(k, args) for k in CITIES}
        print("\nPodsumowanie:", {CITIES[k]["label"]: ("OK" if v else "brak plików")
                                  for k, v in wyniki.items()})
    elif args.city:
        run_city(args.city, args)
    else:
        parser.error("podaj --city <miasto> albo --all")


if __name__ == "__main__":
    main()