"""
wcs_fetch.py - pobieranie MALEGO kadru NMPT z uslugi WCS GUGiK dla wybranego punktu.

* kadr sklada sie z kafelkow 256x256 px (128 m przy 0.5 m/px) - wieksze zapytania
  przekraczaly limit czasu serwera,
* kafelki sa przyciete do GLOBALNEJ siatki 128 m w EPSG:2180, dzieki czemu cache na dysku
  jest wspolny dla wszystkich uzytkownikow (ten sam kafelek pobieramy raz),
* kafelki pobieraja sie rownolegle (workers) - to cudza usluga, wiec z umiarem.

Predkosc serwisu GUGiK zmierzono u uzytkownika: ok. 20 s na kafelek jednym zapytaniem.
Wplyw rownoleglosci trzeba zmierzyc na realnym serwerze.
"""
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests

WCS_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMPT/GRID1/WCS/DigitalSurfaceModel"
COVERAGE = "DSM_PL-KRON86-NH"      # NMPT, wysokosci Kronsztadt 86
# Warstwy (adresy i nazwy z oficjalnej strony geoportal.gov.pl -> Uslugi pobierania WCS):
#   nmpt - model pokrycia terenu (dachy, korony drzew), Kronsztadt 86
#   nmt  - model samego terenu (grunt bez budynkow i drzew), Arc/Info ASCII Grid, EVRF2007
NMT_URL = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/GRID1/WCS/DigitalTerrainModel"
NMT_COVERAGE = "DTM_PL-EVRF2007-NH"
LAYERS = {
    "nmpt": {"url": WCS_URL, "coverage": COVERAGE},
    "nmt": {"url": NMT_URL, "coverage": NMT_COVERAGE},
}
RES = 0.5                          # m/px
TILE_PX = 256
TILE_M = TILE_PX * RES             # 128 m
AAIGRID_KEYS = {"ncols", "nrows", "xllcorner", "xllcenter", "yllcorner", "yllcenter",
                "cellsize", "dx", "dy", "nodata_value"}

# --- GRS80 / PL-1992 (EPSG:2180), bez zaleznosci od pyproj ---
_A, _F = 6378137.0, 1 / 298.257222101
_K0, _L0, _FE, _FN = 0.9993, math.radians(19.0), 500000.0, -5300000.0
_N = _F / (2 - _F)
_AA = _A / (1 + _N) * (1 + _N**2 / 4 + _N**4 / 64)
_E = math.sqrt(_F * (2 - _F))
_ALPHA = (_N / 2 - 2 * _N**2 / 3 + 5 * _N**3 / 16, 13 * _N**2 / 48 - 3 * _N**3 / 5, 61 * _N**3 / 240)
_BETA = (_N / 2 - 2 * _N**2 / 3 + 37 * _N**3 / 96, _N**2 / 48 + _N**3 / 15, 17 * _N**3 / 480)
_DELTA = (2 * _N - 2 * _N**2 / 3 - 2 * _N**3, 7 * _N**2 / 3 - 8 * _N**3 / 5, 56 * _N**3 / 15)


def latlon_to_pl1992(lat, lon):
    phi, dl = math.radians(lat), math.radians(lon) - _L0
    t = math.sinh(math.atanh(math.sin(phi)) - _E * math.atanh(_E * math.sin(phi)))
    xi_p = math.atan2(t, math.cos(dl))
    eta_p = math.atanh(math.sin(dl) / math.sqrt(1 + t * t))
    xi = xi_p + sum(_ALPHA[j - 1] * math.sin(2 * j * xi_p) * math.cosh(2 * j * eta_p) for j in (1, 2, 3))
    eta = eta_p + sum(_ALPHA[j - 1] * math.cos(2 * j * xi_p) * math.sinh(2 * j * eta_p) for j in (1, 2, 3))
    return _FE + _K0 * _AA * eta, _FN + _K0 * _AA * xi          # (E, N)


def pl1992_to_latlon(e, n):
    xi, eta = (n - _FN) / (_K0 * _AA), (e - _FE) / (_K0 * _AA)
    xi_p = xi - sum(_BETA[j - 1] * math.sin(2 * j * xi) * math.cosh(2 * j * eta) for j in (1, 2, 3))
    eta_p = eta - sum(_BETA[j - 1] * math.cos(2 * j * xi) * math.sinh(2 * j * eta) for j in (1, 2, 3))
    chi = math.asin(math.sin(xi_p) / math.cosh(eta_p))
    lat = chi + sum(_DELTA[j - 1] * math.sin(2 * j * chi) for j in (1, 2, 3))
    lon = _L0 + math.atan2(math.sinh(eta_p), math.cos(xi_p))
    return math.degrees(lat), math.degrees(lon)


# --- WCS ---
def parse_aaigrid(text, h, w):
    tok = text.split()
    head, i = {}, 0
    while i + 1 < len(tok) and tok[i].lower() in AAIGRID_KEYS:
        head[tok[i].lower()] = tok[i + 1]
        i += 2
    nrows, ncols = int(head["nrows"]), int(head["ncols"])
    nodata = float(head.get("nodata_value", -9999))
    arr = np.array(tok[i:i + nrows * ncols], dtype=np.float32).reshape(nrows, ncols)
    arr[arr == nodata] = np.nan
    if (nrows, ncols) != (h, w):
        ri = np.minimum((np.arange(h) * nrows / h).astype(int), nrows - 1)
        ci = np.minimum((np.arange(w) * ncols / w).astype(int), ncols - 1)
        arr = arr[np.ix_(ri, ci)]
    return arr


def fetch_tile(min_e, min_n, retries=4, timeout=180, layer="nmpt"):
    """Jeden kafelek 256x256 px o lewym-dolnym rogu (min_e, min_n) w EPSG:2180."""
    cfg = LAYERS[layer]
    params = {
        "SERVICE": "WCS", "VERSION": "1.0.0", "REQUEST": "GetCoverage",
        "COVERAGE": cfg["coverage"], "FORMAT": "image/x-aaigrid",
        "CRS": "EPSG:2180", "RESPONSE_CRS": "EPSG:2180",
        "BBOX": f"{min_e:.3f},{min_n:.3f},{min_e + TILE_M:.3f},{min_n + TILE_M:.3f}",
        "WIDTH": TILE_PX, "HEIGHT": TILE_PX,
    }
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(cfg["url"], params=params, timeout=timeout)
            r.raise_for_status()
            if r.content[:1] == b"<":               # XML = blad serwisu
                raise RuntimeError(r.text[:200].replace("\n", " "))
            return parse_aaigrid(r.text, TILE_PX, TILE_PX)
        except Exception as exc:
            last = exc
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError(f"Geoportal nie zwrócił kafelka po {retries} próbach: {last}")


def crop_geometry(lat, lon, n_tiles):
    """Lewy-gorny naroznik kadru n_tiles x n_tiles kafelkow, przyciety do siatki 128 m."""
    e, n = latlon_to_pl1992(lat, lon)
    side = n_tiles * TILE_M
    e0 = round((e - side / 2) / TILE_M) * TILE_M
    n1 = round((n + side / 2) / TILE_M) * TILE_M
    return e0, n1, side


def fetch_crop(lat, lon, n_tiles, cache_dir, workers=4, progress=None, layer="nmpt"):
    """
    Zwraca (teren float32 [n*256, n*256], info). Kafelki z cache lub pobrane rownolegle.
    progress(i, n) wolane po kazdym gotowym kafelku.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    e0, n1, side = crop_geometry(lat, lon, n_tiles)
    jobs = []
    for ty in range(n_tiles):
        for tx in range(n_tiles):
            min_e = e0 + tx * TILE_M
            min_n = n1 - (ty + 1) * TILE_M
            jobs.append((ty, tx, min_e, min_n))

    def load(job):
        ty, tx, min_e, min_n = job
        f = cache / f"{layer}_{int(round(min_e))}_{int(round(min_n))}.npy"
        if f.exists():
            try:
                return job, np.load(f), True
            except Exception:
                f.unlink(missing_ok=True)
        arr = fetch_tile(min_e, min_n, layer=layer)
        tmp = f.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        tmp.replace(f)
        return job, arr, False

    out = np.empty((n_tiles * TILE_PX, n_tiles * TILE_PX), dtype=np.float32)
    done = cached = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = [pool.submit(load, j) for j in jobs]
        for fut in as_completed(futures):
            (ty, tx, _, _), arr, was_cached = fut.result()
            out[ty * TILE_PX:(ty + 1) * TILE_PX, tx * TILE_PX:(tx + 1) * TILE_PX] = arr
            done += 1
            cached += int(was_cached)
            if progress:
                progress(done, len(jobs))

    nan_frac = float(np.isnan(out).mean())
    if nan_frac > 0.5 or not np.isfinite(out).any():
        raise ValueError(f"Geoportal nie ma danych ({layer.upper()}) dla tego miejsca "
                         "(poza zasięgiem lub obszar bez pomiaru). Wybierz inne miejsce.")
    out = np.where(np.isnan(out), np.nanmedian(out), out).astype(np.float32)
    return out, {"e0": e0, "n1": n1, "side_m": side, "tiles": len(jobs),
                 "from_cache": cached, "nan_frac": nan_frac}


def cleanup_cache(cache_dir, max_age_days=7):
    """Kasuje stare kafelki z cache."""
    cache = Path(cache_dir)
    if not cache.exists():
        return
    limit = time.time() - max_age_days * 86400
    for f in list(cache.glob("nmpt_*.npy")) + list(cache.glob("nmt_*.npy")):
        try:
            if f.stat().st_mtime < limit:
                f.unlink()
        except OSError:
            pass