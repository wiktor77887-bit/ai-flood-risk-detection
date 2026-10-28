"""
Animacja_Powodz_Poznan.py
─────────────────────────
UŻYCIE:
  python Animacja.py --spatial --grid 15 --river         ← zalecane, centrum na Warcie
  python Animacja.py --spatial --grid 21 --river         ← jeszcze większy obszar
  python Animacja.py --spatial --grid 15 --center 1400   ← ręczne centrum
  python Animacja.py --full --downsample 3               ← cały Poznań (wolniejsze)
  python Animacja.py --spatial --grid 15 --river --fps 6 ← szybsza animacja

WYMAGA:
  NMPT_POZNAN_CALOSC.tif
  idealna_maska_budynkow.tif

OPCJONALNIE (dla --river):
  pip install pyproj       ← precyzyjne centrowanie na Warcie
  (bez pyproj: auto-centrum na największym skupisku budynków)
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
from scipy.ndimage import label as nd_label
import rasterio

GAUGE_BASE_M = 49.46
PATCH        = 256

FLOOD_EVENTS = [
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


def gauge_to_masl(gauge_cm, base=None):
    return (base if base is not None else GAUGE_BASE_M) + gauge_cm / 100.0


def nearest_event_label(water_level, levels, names, colors):
    closest = min(range(len(levels)), key=lambda i: abs(levels[i] - water_level))
    return names[closest], colors[closest]


# ==========================================
# POWÓDŹ — FLOOD FILL OD KORYTA RZEKI
# ==========================================
def build_river_seed(terrain, gauge_base):
    """
    Seed powodzi = koryto rzeki dotykające krawędzi mapy + duże zbiorniki wodne.
    Dzięki temu woda zalewa tylko obszary połączone z Wartą,
    a nie każdą nieckę czy bajorko na mapie.
    """
    # 1. Koryto rzeki — niski próg, tylko elementy dotykające krawędzi
    river_level = gauge_base + 0.30
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

    # 2. Duże zbiorniki wodne (Jez. Maltańskie, starorzecza itp.)
    MIN_LAKE_PX = 50000   # ~0.3 km² przy 0.5m/px — wyklucza bajorka i parki
    lake_level  = gauge_base + 1.50
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


def compute_flood_mask(terrain, water_level, river_seed):
    """Zalewa tylko piksele połączone z korytem rzeki."""
    below      = terrain < water_level
    candidates = below | river_seed
    labeled, _ = nd_label(candidates)

    river_labels = set(labeled[river_seed].ravel())
    river_labels.discard(0)

    flooded = np.zeros(terrain.shape, dtype=bool)
    for lbl in river_labels:
        flooded |= (labeled == lbl)

    return flooded & (below | river_seed)


def precompute_all_floods(terrain, water_levels_unique, river_seed):
    flood_cache = {}
    n = len(water_levels_unique)
    for i, wl in enumerate(sorted(water_levels_unique)):
        flood_cache[wl] = compute_flood_mask(terrain, wl, river_seed)
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


def get_spatial_crop(terrain, mask, center_patch=None, grid_size=15,
                     river_center=False, tif_path=None):
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
            from pyproj import Transformer
            with rasterio.open(tif_path) as src:
                trf = src.transform
                crs = src.crs
            tr  = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
            # Punkt między Mostem Rocha a Jez. Maltańskim
            rx, ry = tr.transform(16.9480, 52.4054)
            cx = int((rx - trf.c) / trf.a)
            cy = int((ry - trf.f) / trf.e)
            cx = max(crop_px // 2, min(w - crop_px // 2, cx))
            cy = max(crop_px // 2, min(h - crop_px // 2, cy))
            print(f"  Centrum Warta (pyproj) → piksel ({cx}, {cy})")
            placed = True
        except ImportError:
            print("  pyproj niedostępny — szukam najniższego obszaru na krawędzi")
        except Exception as e:
            print(f"  pyproj błąd: {e} — szukam automatycznie")

        # --- Fallback bez pyproj: znajdź gdzie rzeka wchodzi w kadr ---
        if not placed:
            # Warta wchodzi z lewej lub dołu — szukaj najniższego punktu na krawędzi
            river_level  = float(np.percentile(terrain, 2)) + 1.0
            left_col     = terrain[:, :crop_px // 2]
            bottom_row   = terrain[max(0, h - crop_px // 2):, :]

            # Znajdź wiersz gdzie rzeka wchodzi z lewej
            left_river = left_col < river_level
            if left_river.any():
                river_rows = np.where(left_river.any(axis=1))[0]
                cy = int(np.median(river_rows))
                cx = crop_px // 2
            else:
                # Środek mapy z największą ilością budynków
                best_sum = -1.0
                margin   = PATCH
                for yy in range(margin, max(margin + 1, h - crop_px - margin), PATCH):
                    for xx in range(margin, max(margin + 1, w - crop_px - margin), PATCH):
                        s = float(mask[yy:yy + crop_px, xx:xx + crop_px].sum())
                        if s > best_sum:
                            best_sum = s
                            cy = yy + crop_px // 2
                            cx = xx + crop_px // 2
            cx = max(crop_px // 2, min(w - crop_px // 2, cx))
            cy = max(crop_px // 2, min(h - crop_px // 2, cy))
            print(f"  Auto centrum (fallback) → piksel ({cx}, {cy})")

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
def make_animation(terrain, mask, output_path, title,
                   gauge_base=None, fps=4, dpi=None):

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

    levels = [gauge_to_masl(e["gauge_cm"], gb) for e in FLOOD_EVENTS]
    names  = [
        f'{e["name"]} ({e["gauge_cm"]} cm)' + (f'  [{e["year"]}]' if e["year"] else "")
        for e in FLOOD_EVENTS
    ]
    colors = [e["color"] for e in FLOOD_EVENTS]

    # Sekwencja klatek — płynne przejście + zatrzymanie na kluczowych poziomach
    water_seq = []
    for i in range(len(levels) - 1):
        for t in np.linspace(0, 1, 10, endpoint=False):
            water_seq.append(round(levels[i] + t * (levels[i + 1] - levels[i]), 4))
        for _ in range(6):
            water_seq.append(round(levels[i + 1], 4))
    for _ in range(12):
        water_seq.append(round(levels[-1], 4))

    print(f"  Klatek: {len(water_seq)}")

    print("  Wyznaczam koryto rzeki i zbiorniki...")
    river_seed = build_river_seed(terrain, gb)

    unique_levels = sorted(set(water_seq))
    print(f"  Pre-obliczam {len(unique_levels)} masek powodzi...")
    flood_cache = precompute_all_floods(terrain, unique_levels, river_seed)
    print("  Gotowe. Składam klatki...")

    bld_mask  = mask > 0.5
    bld_total = int(bld_mask.sum())

    frame_data = []
    for wl in water_seq:
        flooded  = flood_cache[wl]
        vis_bld  = bld_mask & (~flooded)
        fld_bld  = bld_mask & flooded
        lbl, col = nearest_event_label(wl, levels, names, colors)
        frame_data.append((flooded, vis_bld, fld_bld, lbl, col,
                            (wl - gb) * 100, wl))

    # Kolormapa terenu — beżowo-brązowa żeby budynki (żółte) i woda (niebieska)
    # były wyraźnie widoczne i nie zlewały się z tłem
    terrain_cmap = LinearSegmentedColormap.from_list(
        "urban", [
            '#1a0a00',   # koryto rzeki — ciemny brąz
            '#5c3a1e',   # nisko        — brąz
            '#9b7540',   # średnio      — piasek
            '#c8aa72',   # wyżej        — jasny piasek
            '#d4c49a',   # jeszcze wyżej — kremowy
            '#e8e0c8',   # wysoko       — jasnobeżowy
        ]
    )

    aspect  = terrain.shape[1] / terrain.shape[0]
    fig_w   = min(9 * aspect, 20)
    fig, ax = plt.subplots(figsize=(fig_w, 9))
    fig.patch.set_facecolor('#0D1B2A')
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)

    t_min = float(np.percentile(terrain, 1))
    t_max = float(np.percentile(terrain, 99))

    ax.axis('off')
    ax.set_facecolor('#0D1B2A')

    # Warstwy imshow — aktualizowane przez set_data() zamiast clear()
    # Dzięki temu animacja jest szybka (blit=True)
    im_terrain = ax.imshow(terrain, cmap=terrain_cmap, vmin=t_min, vmax=t_max,
                            interpolation='bilinear', aspect='equal')
    river_always = terrain < (gauge_base + 1.5)
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

    safe_rgba  = np.zeros((*terrain.shape, 4), dtype=np.float32)
    im_safe    = ax.imshow(safe_rgba, interpolation='nearest', aspect='equal')

    flood_rgba = np.zeros((*terrain.shape, 4), dtype=np.float32)
    im_flood   = ax.imshow(flood_rgba, interpolation='nearest', aspect='equal')

    fs = 12
    txt_water = ax.text(
        0.015, 0.975, "", transform=ax.transAxes,
        color='white', fontsize=fs, fontweight='bold', va='top',
        fontfamily='monospace',
        bbox=dict(facecolor='#0A1628', alpha=0.88,
                  edgecolor='white', linewidth=2, boxstyle='round,pad=0.5'))

    txt_event = ax.text(
        0.015, 0.865, "", transform=ax.transAxes,
        color='white', fontsize=fs - 1, fontweight='bold', va='top',
        bbox=dict(facecolor='#0A1628', alpha=0.88,
                  edgecolor='white', linewidth=1.5, boxstyle='round,pad=0.4'))

    txt_stats = ax.text(
        0.985, 0.975, "", transform=ax.transAxes,
        color='white', fontsize=fs - 1, va='top', ha='right',
        fontfamily='monospace',
        bbox=dict(facecolor='#0A1628', alpha=0.88,
                  edgecolor='#37474F', linewidth=1, boxstyle='round,pad=0.5'))

    ax.legend(
        handles=[
            mpatches.Patch(facecolor='#0A52D4', alpha=0.85, label='Woda'),
            mpatches.Patch(facecolor='#FFE000', label='Budynki - ocalałe'),
            mpatches.Patch(facecolor='#FF1414', label='Budynki - zalane'),
        ],
        loc='lower right', fontsize=fs - 2,
        facecolor='#0A1628', edgecolor='#37474F',
        labelcolor='white', framealpha=0.9
    )
    ax.set_title(title, color='white', fontsize=14, fontweight='bold', pad=8)

    def update(fi):
        flooded, vis_bld, fld_bld, lbl, evt_color, gauge_cm, wl = frame_data[fi]
        bld_flooded = int(fld_bld.sum())

        # Woda — alpha zależy od głębokości
        water_depth = np.where(flooded, wl - terrain, 0.0)
        max_d       = max(float(water_depth.max()), 0.01)
        water_rgba[..., 3] = (
            np.clip(water_depth / max_d * 0.37 + 0.55, 0.0, 0.92)
            * flooded.astype(np.float32)
        )
        im_water.set_data(water_rgba)

        # Budynki ocalałe — żółte, alpha=1
        safe_rgba[...]        = 0.0
        safe_rgba[vis_bld, 0] = 1.0
        safe_rgba[vis_bld, 1] = 0.88
        safe_rgba[vis_bld, 2] = 0.0
        safe_rgba[vis_bld, 3] = 1.0
        im_safe.set_data(safe_rgba)

        # Budynki zalane — czerwone, alpha=1
        flood_rgba[...]        = 0.0
        flood_rgba[fld_bld, 0] = 1.0
        flood_rgba[fld_bld, 1] = 0.10
        flood_rgba[fld_bld, 2] = 0.0
        flood_rgba[fld_bld, 3] = 1.0
        im_flood.set_data(flood_rgba)

        txt_water.set_text(
            f"Poziom wody:  {wl:.2f} m n.p.m.\n"
            f"Wodowskaz:    {gauge_cm:.0f} cm"
        )
        txt_water.get_bbox_patch().set_edgecolor(evt_color)

        txt_event.set_text(lbl)
        txt_event.set_color(evt_color)
        txt_event.get_bbox_patch().set_edgecolor(evt_color)

        txt_stats.set_text(
            f"Zalany teren:    {flooded.sum()/terrain.size*100:5.1f}%\n"
            f"Zalane budynki: {bld_flooded/max(bld_total,1)*100:5.1f}%\n"
            f"  ({bld_flooded:,} / {bld_total:,} px)"
        )

        return [im_water, im_safe, im_flood, txt_water, txt_event, txt_stats]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frame_data),
        blit=True, interval=1000 // fps
    )

    print(f"Zapisywanie {output_path} ...")
    ani.save(output_path, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)

    size_mb = Path(output_path).stat().st_size / 1024**2
    print(f"✅ Zapisano: {output_path}  ({size_mb:.1f} MB)")


# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser(
        description="Animacja powodzi Warty — Poznań",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Przykłady:
  python Animacja.py --spatial --grid 15 --river
  python Animacja.py --spatial --grid 21 --river --fps 6
  python Animacja.py --spatial --grid 15 --center 1400
  python Animacja.py --full --downsample 3
        """
    )
    parser.add_argument("--tif",        default="NMPT_POZNAN_CALOSC.tif")
    parser.add_argument("--mask_tif",   default="idealna_maska_budynkow.tif")
    parser.add_argument("--spatial",    action="store_true",
                        help="Kadr przestrzenny grid×grid kafelków")
    parser.add_argument("--full",       action="store_true",
                        help="Cały Poznań (użyj z --downsample)")
    parser.add_argument("--river",      action="store_true",
                        help="Centruj na Warcie (wymaga pyproj lub auto-fallback)")
    parser.add_argument("--center",     type=int,   default=None,
                        help="Numer kafelka jako centrum kadru")
    parser.add_argument("--grid",       type=int,   default=15,
                        help="Rozmiar siatki N×N kafelków (domyślnie 15)")
    parser.add_argument("--downsample", type=int,   default=1,
                        help="Próbkowanie co N pikseli (dla --full)")
    parser.add_argument("--fps",        type=int,   default=4)
    parser.add_argument("--dpi",        type=int,   default=None,
                        help="Rozdzielczość GIF (domyślnie auto)")
    parser.add_argument("--out",        default="powodz_warta_poznan.gif")
    args = parser.parse_args()

    for f in [args.tif, args.mask_tif]:
        if not Path(f).exists():
            print(f"❌ Nie znaleziono: {f}")
            print("   Uruchom skrypt z folderu projektu gdzie są pliki .tif")
            return

    ds = args.downsample if args.full else 1

    print("=" * 58)
    print("  Symulacja powodzi Warty — Poznań")
    print("=" * 58)

    print("\nWczytuję teren...")
    terrain = load_tif(args.tif,      is_mask=False, downsample=ds)
    print("Wczytuję maskę budynków...")
    mask    = load_tif(args.mask_tif, is_mask=True,  downsample=ds)
    print("Wyrównuję rozmiary...")
    terrain, mask = align_arrays(terrain, mask)

    # Kalibracja poziomu wody na podstawie rzeczywistego dna koryta
    river_bed  = float(np.percentile(terrain, 1))
    gauge_base = river_bed
    print(f"\n  [Kalibracja] dno koryta ~ {river_bed:.2f} m n.p.m.")
    print(f"  {'Zdarzenie':25s} {'cm':>5}   {'m n.p.m.':>10}   {'zalanie':>7}")
    print("  " + "─" * 55)
    for e in FLOOD_EVENTS:
        lvl = gauge_base + e["gauge_cm"] / 100.0
        pct = float((terrain < lvl).mean()) * 100
        print(f"  {e['name']:25s} {e['gauge_cm']:5d}   {lvl:10.2f}   {pct:6.1f}%")
    print()

    if args.full:
        title = f"Symulacja powodzi Warty — Poznań (pełna mapa, 1/{ds})"
    else:
        print(f"Tryb: przestrzenny {args.grid}×{args.grid} kafelków "
              f"({'--river' if args.river else '--center' if args.center else 'auto'})")
        terrain, mask = get_spatial_crop(
            terrain, mask,
            center_patch=args.center,
            grid_size=args.grid,
            river_center=args.river,
            tif_path=args.tif
        )
        km = args.grid * PATCH * 0.5 / 1000
        title = (f"Symulacja powodzi Warty — Poznań  "
                 f"({args.grid}×{args.grid} kafelków, ~{km:.0f}×{km:.0f} km)")

    print()
    make_animation(
        terrain, mask, args.out,
        title=title, gauge_base=gauge_base,
        fps=args.fps, dpi=args.dpi
    )
    print(f"\n🎉 Gotowe! Otwórz: {args.out}")


if __name__ == "__main__":
    main()