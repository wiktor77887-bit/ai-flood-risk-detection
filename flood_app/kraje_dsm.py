"""
kraje_dsm.py - maly kadr NMPT (DSM) z zagranicznych geoportali dla wybranego punktu.

Dziala jak wcs_fetch.py dla Polski: kadr sklada sie z kafelkow 128 x 128 m (256 x 256 px przy 0,5 m/px),
przycietych do globalnej siatki 128 m w ukladzie danego kraju, wiec cache na dysku jest wspolny dla
wszystkich uzytkownikow. Zrodla (sprawdzone skryptem sprawdz_zrodla.py):

  NL  Holandia   AHN, WCS PDOK (coverage dsm_05m, EPSG:28992, 0,5 m)
  CH  Szwajcaria swissSURFACE3D Raster, katalog STAC swisstopo -> kafelki COG 1 x 1 km (EPSG:2056, 0,5 m),
                 czytane fragmentami przez /vsicurl/
  GB  Anglia     Environment Agency, LIDAR Composite First Return DSM, WCS (EPSG:27700, 1 m -> 0,5 m)

Zwracany kadr jest zawsze w 0,5 m/px i w metrach n.p.m., tak jak z wcs_fetch.fetch_crop.
"""
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np
import requests

for _k, _v in {"GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
               "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff",
               "GDAL_HTTP_TIMEOUT": "60", "GDAL_HTTP_MAX_RETRY": "3", "VSI_CACHE": "TRUE"}.items():
    os.environ.setdefault(_k, _v)

import rasterio
from rasterio.io import MemoryFile
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import transform as _warp

TILE_M = 128.0
TILE_PX = 256
RES = 0.5
HEAD = {"User-Agent": "flood-app/1.0"}

STAC_ITEMS = "https://data.geo.admin.ch/api/stac/v1/collections/ch.swisstopo.swisssurface3d-raster/items"

KRAJE = {
    "PL": {"nazwa": "Polska", "zrodlo": "GUGiK (geoportal.gov.pl)"},
    "NL": {
        "nazwa": "Holandia", "epsg": 28992, "typ": "wcs",
        "url": "https://service.pdok.nl/rws/ahn/wcs/v1_0", "coverage_zawiera": "dsm", "osie": ("x", "y"),
        "zasieg": (3.2, 50.7, 7.3, 53.6),            # lon_min, lat_min, lon_max, lat_max
        "zrodlo": "AHN, Rijkswaterstaat (PDOK), licencja CC0",
        "woda_nodata": True,      # AHN nie ma pomiaru nad woda: kanal/rzeka to dziura w danych (NaN)
        "uwaga": "W AHN woda nie ma pomiaru (jest dziurą w danych), więc jej poziom jest szacowany "
                 "jako najniższy teren w kadrze. Wynik jest orientacyjny.",
        "miasta": {"Amsterdam": (52.3730, 4.8990), "Rotterdam": (51.9225, 4.4792), "Haga": (52.0705, 4.3007),
                   "Utrecht": (52.0900, 5.1180),
                   "Arnhem (Ren)": (51.9810, 5.9070), "Maastricht (Moza)": (50.8480, 5.6890)},
    },
    "CH": {
        "nazwa": "Szwajcaria", "epsg": 2056, "typ": "stac",
        "zasieg": (5.9, 45.8, 10.5, 47.9),
        "zrodlo": "swissSURFACE3D Raster, swisstopo (Open Government Data)",
        "uwaga": "",
        "miasta": {"Zurych": (47.3769, 8.5417), "Berno (Aare)": (46.9480, 7.4474), "Genewa": (46.2044, 6.1432),
                   "Bazylea (Ren)": (47.5590, 7.5880), "Lucerna": (47.0500, 8.3090)},
    },
    "GB": {
        "nazwa": "Anglia", "epsg": 27700, "typ": "wcs", "res_zrodla": 1.0,
        "url": "https://environment.data.gov.uk/spatialdata/lidar-composite-digital-surface-model-first-return-dsm-1m/wcs",
        "coverage_zawiera": "elevation", "osie": ("E", "N"),
        "zasieg": (-6.5, 49.8, 1.9, 55.9),
        "zrodlo": "Environment Agency, LIDAR Composite First Return DSM 1 m, © Crown copyright, Open Government Licence v3.0",
        "uwaga": "Dane mają rozdzielczość 1 m (model uczono na 0,5 m), więc budynki mogą być wykrywane mniej dokładnie. "
                 "Dotyczy tylko Anglii (nie całej Wielkiej Brytanii).",
        "miasta": {"Londyn (Tamiza)": (51.5074, -0.1278), "Birmingham": (52.4862, -1.8904),
                   "Manchester": (53.4808, -2.2426), "Newcastle (Tyne)": (54.9690, -1.6040),
                   "York (Ouse)": (53.9590, -1.0800), "Bristol": (51.4500, -2.5990), "Oxford": (51.7520, -1.2570)},
    },
}


def kraje_zagraniczne():
    return [k for k in KRAJE if k != "PL"]


def w_zasiegu(kod, lat, lon):
    z = KRAJE[kod].get("zasieg")
    return bool(z) and z[0] <= lon <= z[2] and z[1] <= lat <= z[3]


# ---------------------------------------------------------------- geometria
def latlon_do_ukladu(kod, lat, lon):
    x, y = _warp("EPSG:4326", f"EPSG:{KRAJE[kod]['epsg']}", [lon], [lat])
    return x[0], y[0]


def uklad_do_latlon(kod, x, y):
    lon, lat = _warp(f"EPSG:{KRAJE[kod]['epsg']}", "EPSG:4326", [x], [y])
    return lat[0], lon[0]


def crop_geometry(kod, lat, lon, n_tiles):
    """Lewy-gorny naroznik kadru n x n kafelkow, przyciety do siatki 128 m w ukladzie kraju."""
    x, y = latlon_do_ukladu(kod, lat, lon)
    side = n_tiles * TILE_M
    x0 = round((x - side / 2) / TILE_M) * TILE_M
    y1 = round((y + side / 2) / TILE_M) * TILE_M
    return x0, y1, side


def rogi_latlon(kod, x0, y1, side):
    pts = [(x0, y1), (x0 + side, y1), (x0 + side, y1 - side), (x0, y1 - side)]
    return [uklad_do_latlon(kod, x, y) for x, y in pts]


# ---------------------------------------------------------------- czytanie rastra z bajtow / pliku
def _tiff_do_tablicy(dane, shape=(TILE_PX, TILE_PX)):
    """Bajty GeoTIFF -> float32 (NaN = brak danych), przeskalowane do shape (bilinear)."""
    with MemoryFile(dane) as mf, mf.open() as src:
        a = src.read(1, out_shape=shape, resampling=Resampling.bilinear, masked=True)
    return _czysc(a)


def _czysc(a):
    a = np.ma.filled(a.astype(np.float32), np.nan) if np.ma.isMaskedArray(a) else a.astype(np.float32)
    a[(a < -100) | (a > 9000)] = np.nan
    return a


# ---------------------------------------------------------------- WCS 2.0.1 (NL, GB)
@lru_cache(maxsize=8)
def _coverage_id(kod):
    cfg = KRAJE[kod]
    r = requests.get(cfg["url"], params={"SERVICE": "WCS", "VERSION": "2.0.1", "REQUEST": "GetCapabilities"},
                     headers=HEAD, timeout=60)
    r.raise_for_status()
    ids = re.findall(r"<(?:\w+:)?CoverageId>([^<]+)<", r.text)
    for i in ids:
        if cfg["coverage_zawiera"] in i.lower():
            return i
    raise RuntimeError(f"Usługa WCS ({cfg['nazwa']}) nie podaje spodziewanej warstwy "
                       f"(szukano '{cfg['coverage_zawiera']}' wśród: {', '.join(ids[:6])}).")


def _kafelek_wcs(kod, x0, y0, retries=3, timeout=120):
    cfg = KRAJE[kod]
    ax, ay = cfg["osie"]
    epsg = cfg["epsg"]
    crs = f"http://www.opengis.net/def/crs/EPSG/0/{epsg}"
    # pare pikseli zapasu z kazdej strony nie jest potrzebne: serwer zwraca zakres dokladnie wg SUBSET
    params = [("SERVICE", "WCS"), ("VERSION", "2.0.1"), ("REQUEST", "GetCoverage"),
              ("COVERAGEID", _coverage_id(kod)),
              ("SUBSET", f"{ax}({x0:.3f},{x0 + TILE_M:.3f})"), ("SUBSET", f"{ay}({y0:.3f},{y0 + TILE_M:.3f})"),
              ("SUBSETTINGCRS", crs), ("OUTPUTCRS", crs), ("FORMAT", "image/tiff")]
    last = None
    for proba in range(retries):
        try:
            r = requests.get(cfg["url"], params=params, headers=HEAD, timeout=timeout)
            r.raise_for_status()
            if r.content[:4] not in (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"):
                raise RuntimeError(r.text[:200].replace("\n", " "))
            return _tiff_do_tablicy(r.content)
        except Exception as exc:
            last = exc
            time.sleep(2 ** (proba + 1))
    raise RuntimeError(f"{cfg['nazwa']}: nie udało się pobrać kafelka po {retries} próbach: {last}")


# ---------------------------------------------------------------- STAC + COG (CH)
def _otworz_zdalny(href):
    """Lokalny plik (testy) albo adres https przez /vsicurl/."""
    return rasterio.open(href if os.path.exists(href) else "/vsicurl/" + href)


def _stac_kafelki(lon_min, lat_min, lon_max, lat_max, retries=3):
    """Adresy kafelkow .tif (0,5 m) dla prostokata WGS84; z kazdego pola 1 km tylko najnowszy rok."""
    params = {"bbox": f"{lon_min:.6f},{lat_min:.6f},{lon_max:.6f},{lat_max:.6f}", "limit": 20}
    last = None
    for proba in range(retries):
        try:
            r = requests.get(STAC_ITEMS, params=params, headers=HEAD, timeout=60)
            r.raise_for_status()
            wybrane = {}
            for it in r.json().get("features", []):
                m = re.search(r"_(\d{4})_(\d+-\d+)$", it.get("id", ""))
                rok, pole = (int(m.group(1)), m.group(2)) if m else (0, it.get("id"))
                href = next((a["href"] for k, a in (it.get("assets") or {}).items()
                             if str(a.get("href", "")).lower().endswith(".tif") and "_0.5_" in a["href"]), None)
                if href and (pole not in wybrane or rok > wybrane[pole][0]):
                    wybrane[pole] = (rok, href)
            return [h for _, h in wybrane.values()]
        except Exception as exc:
            last = exc
            time.sleep(2 ** (proba + 1))
    raise RuntimeError(f"Szwajcaria: katalog STAC nie odpowiedział: {last}")


def _kafelek_stac(kod, x0, y0):
    cfg = KRAJE[kod]
    rogi = [uklad_do_latlon(kod, x, y) for x, y in
            ((x0, y0), (x0 + TILE_M, y0), (x0 + TILE_M, y0 + TILE_M), (x0, y0 + TILE_M))]
    lats, lons = [p[0] for p in rogi], [p[1] for p in rogi]
    hrefy = _stac_kafelki(min(lons), min(lats), max(lons), max(lats))
    if not hrefy:
        raise RuntimeError(f"{cfg['nazwa']}: brak danych swissSURFACE3D dla tego miejsca (poza zasięgiem).")
    dsy = [_otworz_zdalny(h) for h in hrefy]
    try:
        mozaika, _ = merge(dsy, bounds=(x0, y0, x0 + TILE_M, y0 + TILE_M), res=RES, nodata=-9999,
                           resampling=Resampling.bilinear)
    finally:
        for d in dsy:
            d.close()
    a = mozaika[0].astype(np.float32)
    a[a == -9999] = np.nan
    if a.shape != (TILE_PX, TILE_PX):
        raise RuntimeError(f"Nieoczekiwany rozmiar kafelka {a.shape}")
    return _czysc(a)


def _kafelek(kod, x0, y0):
    typ = KRAJE[kod]["typ"]
    return _kafelek_wcs(kod, x0, y0) if typ == "wcs" else _kafelek_stac(kod, x0, y0)


# ---------------------------------------------------------------- kadr
def fetch_crop(kod, lat, lon, n_tiles, cache_dir, workers=4, progress=None):
    """Zwraca (teren float32 [n*256, n*256] w 0,5 m, info) jak wcs_fetch.fetch_crop."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    x0, y1, side = crop_geometry(kod, lat, lon, n_tiles)
    jobs = [(ty, tx, x0 + tx * TILE_M, y1 - (ty + 1) * TILE_M) for ty in range(n_tiles) for tx in range(n_tiles)]

    def load(job):
        ty, tx, mx, my = job
        f = cache / f"kraj_{kod}_{int(round(mx))}_{int(round(my))}.npy"
        if f.exists():
            try:
                return job, np.load(f), True
            except Exception:
                f.unlink(missing_ok=True)
        arr = _kafelek(kod, mx, my)
        tmp = f.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(f)
        return job, arr, False

    out = np.empty((n_tiles * TILE_PX, n_tiles * TILE_PX), dtype=np.float32)
    done = cached = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(load, j) for j in jobs]
        for fut in as_completed(futures):
            (ty, tx, _, _), arr, z_cache = fut.result()
            out[ty * TILE_PX:(ty + 1) * TILE_PX, tx * TILE_PX:(tx + 1) * TILE_PX] = arr
            done += 1
            cached += int(z_cache)
            if progress:
                progress(done, len(jobs))

    nan_frac = float(np.isnan(out).mean())
    woda = bool(KRAJE[kod].get("woda_nodata"))
    if nan_frac > (0.9 if woda else 0.5) or not np.isfinite(out).any():
        raise ValueError(f"{KRAJE[kod]['nazwa']}: brak danych wysokościowych dla tego miejsca "
                         "(poza zasięgiem lub obszar bez pomiaru). Wybierz inne miejsce.")
    if woda:
        out = _wypelnij_wode(out)
    out = np.where(np.isnan(out), np.nanmedian(out), out).astype(np.float32)
    return out, {"x0": x0, "y1": y1, "side_m": side, "tiles": len(jobs), "from_cache": cached,
                 "nan_frac": nan_frac}


def _wypelnij_wode(a, min_px=400):
    """Duze dziury w danych (woda w AHN) -> plaska powierzchnia nieco ponizej najnizszego terenu w kadrze,
    zeby silnik rozpoznal je jako rzeke (plaski, najnizszy obszar). Male dziury zalatwia mediana."""
    from scipy import ndimage as ndi
    nan = np.isnan(a)
    if not nan.any() or nan.all():
        return a
    lab, n = ndi.label(nan)
    rozm = ndi.sum(nan, lab, index=np.arange(1, n + 1))
    duze = np.isin(lab, 1 + np.where(rozm >= min_px)[0])
    if not duze.any():
        return a
    poziom = float(np.nanpercentile(a, 1)) - 0.5
    a = a.copy()
    a[duze] = poziom
    return a


def cleanup_cache(cache_dir, max_age_days=7):
    cache = Path(cache_dir)
    if not cache.exists():
        return
    limit = time.time() - max_age_days * 86400
    for f in cache.glob("kraj_*.npy"):
        try:
            if f.stat().st_mtime < limit:
                f.unlink()
        except OSError:
            pass