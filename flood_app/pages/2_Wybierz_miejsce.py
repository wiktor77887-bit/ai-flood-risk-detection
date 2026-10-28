"""
Strona: symulacja powodzi dla MIEJSCA WYBRANEGO NA MAPIE (bez wgrywania plikow).

Uzytkownik klika punkt przy rzece, wybiera rozmiar (256/384/512 m) i scenariusz.
Serwer pobiera maly kadr NMPT z geoportalu (kafelki 256 px, rownolegle, z cache),
wykrywa budynki modelem U-Net i generuje animacje GIF.
"""
import gc
import inspect
import json
import time
from pathlib import Path

import numpy as np
import streamlit as st

st.set_page_config(page_title="Wybierz miejsce - analiza powodziowa", page_icon="📍", layout="wide")

import folium
from streamlit_folium import st_folium

import flood_engine as fe
import ui_common as ui
import wcs_fetch as wf

ui.styl()
ui.wymagaj_logowania()
ui.nawigacja()
ui.sprzataj()
wf.cleanup_cache(ui.WCS_CACHE)

@st.cache_data(show_spinner=False)
def wczytaj_obrysy():
    """Obrys Polski i granice miast (GUGiK PRG) z dane/obrysy.geojson. Brak pliku = brak warstwy."""
    sciezka = Path(__file__).resolve().parent.parent / "dane" / "obrysy.geojson"
    try:
        return json.loads(sciezka.read_text(encoding="utf-8"))
    except Exception:
        return None


_MIASTA_AWARYJNE = {
    "Poznań": (52.4064, 16.9252), "Warszawa": (52.2297, 21.0122), "Wrocław": (51.1079, 17.0385),
    "Toruń": (53.0138, 18.5984), "Bydgoszcz": (53.1235, 18.0084), "Kraków": (50.0614, 19.9372),
    "Gdańsk": (54.3520, 18.6466),
}
_PL = str.maketrans("ąćęłńóśźżĄĆĘŁŃÓŚŹŻ", "acelnoszzACELNOSZZ")


def _srodek_miasta(f):
    """(lat, lon) miasta: z pliku, a w starszym pliku obrysow - ze srodka jego granicy."""
    p = f["properties"]
    if "lat" in p and "lon" in p:
        return p["lat"], p["lon"]
    if p["nazwa"] in _MIASTA_AWARYJNE:
        return _MIASTA_AWARYJNE[p["nazwa"]]
    g = f["geometry"]
    pierscien = g["coordinates"][0] if g["type"] == "Polygon" else g["coordinates"][0][0]
    return (sum(c[1] for c in pierscien) / len(pierscien), sum(c[0] for c in pierscien) / len(pierscien))


def _miasta():
    """Lista miast do skoku na mapie: z pliku obrysow, a gdy go brak - siedem podstawowych."""
    dane = wczytaj_obrysy()
    if not dane:
        return dict(_MIASTA_AWARYJNE)
    try:
        wynik = {f["properties"]["nazwa"]: _srodek_miasta(f)
                 for f in dane["features"] if f["properties"]["typ"] == "miasto"}
    except Exception:
        return dict(_MIASTA_AWARYJNE)
    if not wynik:
        return dict(_MIASTA_AWARYJNE)
    return dict(sorted(wynik.items(), key=lambda kv: kv[0].translate(_PL).lower()))


MIASTA = _miasta()
ESRI = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"

st.session_state.setdefault("punkt", None)
st.session_state.setdefault("map_center", [52.0, 19.3])
st.session_state.setdefault("map_zoom", 6)


def skocz():
    nazwa = st.session_state.get("miasto_sel")
    if nazwa in MIASTA:
        st.session_state["map_center"] = list(MIASTA[nazwa])
        st.session_state["map_zoom"] = 14


ui.naglowek("📍", "Symulacja dla miejsca",
            "Przybliż mapę do rzeki i kliknij miejsce nad brzegiem. Pobierzemy dane wysokościowe (NMPT) z geoportalu "
            "dla niewielkiego kwadratu wokół punktu i przygotujemy animację zalewania.")
KROKI = ["Miejsce", "Scenariusz", "Wynik"]
pasek_krokow = st.empty()
pasek_krokow.markdown(ui.postep(KROKI, 1 if st.session_state["punkt"] else 0), unsafe_allow_html=True)

ui.krok(1, "Wybierz miejsce", "Kliknij na mapie punkt przy rzece. Obszar jest mały (do 512 m), więc musi obejmować brzeg.")
panel, kol_mapa = st.columns([1.35, 3], gap="large")
with panel:
    with ui.karta("miejsce"):
        st.selectbox("Przejdź do miasta", ["—"] + list(MIASTA), key="miasto_sel", on_change=skocz)
        rozmiar = st.radio("Rozmiar obszaru", [256, 384, 512], index=1, horizontal=True,
                           format_func=lambda m: f"{m} m",
                           help="Większy obszar = więcej kafelków danych do pobrania, więc dłuższe czekanie.")
        n_tiles = rozmiar // 128
        _p = st.session_state["punkt"]
        if _p:
            st.markdown(f"**Wybrany punkt**  \n{_p[0]:.5f}, {_p[1]:.5f}  \nObszar {rozmiar} × {rozmiar} m "
                        f"({n_tiles * n_tiles} kafelków danych)")
        else:
            st.markdown("**Nie wybrano punktu.** Kliknij na mapie miejsce przy rzece.")

# ---- mapa ----
punkt = st.session_state["punkt"]
m = folium.Map(location=st.session_state["map_center"], zoom_start=st.session_state["map_zoom"],
               tiles=ESRI, attr="Esri, Maxar, Earthstar Geographics", attributionControl=False)


LEWO = {"Bydgoszcz", "Gdynia"}      # podpis po lewej stronie kropki, zeby nie zaslaniac sasiada (Torun, Gdansk)


def dodaj_obrysy(mapa):
    dane = wczytaj_obrysy()
    if not dane:
        return
    nieklik = {"interactive": False}          # klik w obrys nie moze blokowac wyboru punktu
    for f in dane["features"]:
        nazwa, typ = f["properties"]["nazwa"], f["properties"]["typ"]
        if typ == "polska":
            for kolor, grub in (("#0F2A3D", 6), ("#FFFFFF", 3)):
                folium.GeoJson(f, style_function=lambda _, k=kolor, g=grub: {
                    "color": k, "weight": g, "fill": False}, **nieklik).add_to(mapa)
        else:
            folium.GeoJson(f, style_function=lambda _: {
                "color": "#FF5A36", "weight": 2.5, "fillColor": "#FF5A36", "fillOpacity": 0.25},
                **nieklik).add_to(mapa)
            lat, lon = MIASTA.get(nazwa) or _srodek_miasta(f)
            folium.CircleMarker((lat, lon), radius=5, color="white", weight=2, fill=True,
                                fill_color="#FF5A36", fill_opacity=1, **nieklik).add_to(mapa)
            ranga = f["properties"].get("ranga", 1)
            lewo = nazwa in LEWO
            przesuniecie = "transform:translateX(calc(-100% - 14px));" if lewo else ""
            html = (f"<div class=\"etk etk{ranga}\" "
                    "style=\"font:700 14px 'Source Sans 3',Arial,sans-serif;color:#fff;"
                    f"{przesuniecie}"
                    "white-space:nowrap;pointer-events:none;"
                    "text-shadow:-1px -1px 2px #0F2A3D,1px -1px 2px #0F2A3D,"
                    "-1px 1px 2px #0F2A3D,1px 1px 2px #0F2A3D,0 0 4px #0F2A3D\">"
                    f"{nazwa}</div>")
            folium.Marker(
                (lat, lon), **nieklik,
                icon=folium.DivIcon(icon_size=(0, 0), icon_anchor=(0, 10) if lewo else (-10, 10),
                                    html=html)).add_to(mapa)
    # mniejsze miasta pokazuja podpis dopiero po przyblizeniu, zeby mapa Polski byla czytelna
    mapa.get_root().script.add_child(folium.Element(
        # po zdarzeniu load: folium tworzy mape PO tym skrypcie, wiec wykonany od razu zepsulby cala mape
        "window.addEventListener('load',function(){var mp=%s;function u(){var z=mp.getZoom();"
        "document.querySelectorAll('.etk2').forEach(function(e){e.style.display=z>=8?'block':'none'});"
        "document.querySelectorAll('.etk3').forEach(function(e){e.style.display=z>=10?'block':'none'});}"
        "mp.on('zoomend',u);mp.on('layeradd',u);u();});" % mapa.get_name()))


dodaj_obrysy(m)
fg = folium.FeatureGroup(name="wybor")
if punkt:
    e0, n1, side = wf.crop_geometry(punkt[0], punkt[1], n_tiles)
    rog = [wf.pl1992_to_latlon(e0, n1), wf.pl1992_to_latlon(e0 + side, n1),
           wf.pl1992_to_latlon(e0 + side, n1 - side), wf.pl1992_to_latlon(e0, n1 - side)]
    folium.Polygon(rog, color=ui.ALERT, weight=3, fill=True, fill_opacity=0.12).add_to(fg)
    folium.CircleMarker(punkt, radius=4, color="white", fill=True, fill_color=ui.ALERT,
                        fill_opacity=1).add_to(fg)

_PARAMS = inspect.signature(st_folium).parameters
HAS_FG = "feature_group_to_add" in _PARAMS
kwargs = {"returned_objects": ["last_clicked"] if HAS_FG else ["last_clicked", "center", "zoom"]}
if HAS_FG:
    kwargs["feature_group_to_add"] = fg      # warstwa dodawana bez resetowania widoku mapy
else:
    fg.add_to(m)

with kol_mapa:
    wyjscie_mapy = st_folium(m, height=560, use_container_width=True, key="mapa_wybor", **kwargs) or {}

if not HAS_FG:                               # starsza wersja: sami pamietamy widok
    if wyjscie_mapy.get("center"):
        st.session_state["map_center"] = [wyjscie_mapy["center"]["lat"], wyjscie_mapy["center"]["lng"]]
    if wyjscie_mapy.get("zoom"):
        st.session_state["map_zoom"] = wyjscie_mapy["zoom"]

klik = wyjscie_mapy.get("last_clicked")
if klik:
    nowy = (round(klik["lat"], 6), round(klik["lng"], 6))
    if nowy != punkt:
        st.session_state["punkt"] = nowy
        st.rerun()

if not punkt:
    st.stop()                       # podpowiedz jest w panelu obok mapy

pasek_krokow.markdown(ui.postep(KROKI, 2), unsafe_allow_html=True)
ui.krok(2, "Scenariusz wzrostu wody", "O ile metrów podnosimy poziom względem rzeki widocznej w NMPT.")
with ui.karta("scenariusz"):
    max_eff, krok = ui.kontrolki_scenariusza()

    uzyj_nmt = st.checkbox(
        "Użyj oficjalnego modelu terenu (NMT) z geoportalu",
        value=False,
        help="Dokładniejszy grunt (bez budynków i drzew), ale pobieranie trwa mniej więcej dwa razy dłużej. "
             "Bez tego grunt jest szacowany z NMPT - zwykle wystarcza, ale bywa mniej dokładny w gęstych "
             "zadrzewieniach i przy dużych budynkach.")

if not ui.model_dostepny():
    st.error("Brak pliku best_model.pth na serwerze, więc nie da się wykryć budynków.")
    st.stop()

st.caption(f"Pobieranie danych z geoportalu trwa zwykle od kilkudziesięciu sekund do kilku minut "
           f"(zapytań naraz: {ui.WCS_WORKERS}){', podwójnie przy NMT' if uzyj_nmt else ''}; "
           "potem generowanie animacji. Nie zamykaj karty. "
           "Dane NMPT: GUGiK (geoportal.gov.pl).")

ui.krok(3, "Generuj animację")
# ---- generowanie ----
sig_biezacy = (punkt, n_tiles, max_eff, krok, uzyj_nmt)

if st.button("Generuj GIF", type="primary"):
    sem, blad = ui.zarezerwuj(rozmiar / 1000, True)
    if blad:
        st.error(blad) if "pamięci" in blad else st.warning(blad)
    else:
        t0 = time.time()
        pasek = st.progress(0.0, text="Start…")
        try:
            czesc = 0.20 if uzyj_nmt else 0.40
            teren, info = wf.fetch_crop(
                punkt[0], punkt[1], n_tiles, ui.WCS_CACHE, workers=ui.WCS_WORKERS,
                progress=lambda i, n: pasek.progress(
                    czesc * i / n, text=f"Pobieram NMPT z geoportalu {i}/{n}"))
            grunt, uwagi = None, []
            if uzyj_nmt:
                try:
                    grunt, info_nmt = wf.fetch_crop(
                        punkt[0], punkt[1], n_tiles, ui.WCS_CACHE, workers=ui.WCS_WORKERS, layer="nmt",
                        progress=lambda i, n: pasek.progress(
                            czesc + czesc * i / n, text=f"Pobieram NMT z geoportalu {i}/{n}"))
                    # NMPT nie powinien lezec wyraznie ponizej gruntu - inaczej dane sa niespojne
                    if float(np.percentile(teren - grunt, 10)) < -2.0:
                        uwagi.append("Pobrany NMT nie pasuje do NMPT (niespójne wysokości), więc użyto "
                                     "gruntu oszacowanego z NMPT.")
                        grunt = None
                except Exception as exc_nmt:
                    uwagi.append(f"Nie udało się pobrać NMT ({exc_nmt}); użyto gruntu oszacowanego z NMPT.")
                    grunt = None
            model, device = ui.zaladuj_model()
            predykcja = ui.funkcja_predykcji(model, device)
            wyjscie = ui.katalog_sesji() / f"miejsce_{int(t0)}.gif"
            start = 0.40 if uzyj_nmt else 0.40
            wynik = fe.generate_from_terrain(
                teren, str(wyjscie), predict_prob=predykcja, ground=grunt,
                max_rise_m=max_eff, step_m=krok, fps=4,
                progress=lambda msg, f: pasek.progress(start + (1 - start) * f, text=msg))
            wynik["notes"] = uwagi + wynik["notes"]
            wynik["from_cache"] = info["from_cache"]
            wynik["tiles"] = info["tiles"]
            st.session_state["wynik_miejsce"] = {"path": str(wyjscie), "info": wynik,
                                                 "t": time.time() - t0, "sig": sig_biezacy}
            pasek.empty()
        except Exception as exc:
            pasek.empty()
            st.error(f"Nie udało się wygenerować animacji: {exc}")
        finally:
            sem.release()
            gc.collect()

w = st.session_state.get("wynik_miejsce")
if w and w["sig"] == sig_biezacy:
    pasek_krokow.markdown(ui.postep(KROKI, 3), unsafe_allow_html=True)
ui.pokaz_wynik(
    w, sig_biezacy, nazwa_pliku="symulacja_powodzi_miejsce.gif",
    dodatek=(f"Kafelki danych z pamięci podręcznej: {w['info']['from_cache']}/{w['info']['tiles']}."
             if w and w["sig"] == sig_biezacy else ""))