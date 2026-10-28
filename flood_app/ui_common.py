"""
ui_common.py - elementy wspolne dla stron generujacych GIF (pages/1_..., pages/2_...).

WAZNE: semafor zadan (sloty) i model sa tu, a nie w kazdej stronie z osobna, bo st.cache_resource
jest kluczowane funkcja - dwie kopie w dwoch stronach dalyby DWA niezalezne semafory i dwa zadania
naraz na serwerze z ograniczona pamiecia.

Zmienne srodowiskowe:
  FLOOD_MAX_KM       najwiekszy bok kadru z pliku w km; bez tej zmiennej limit liczy sie sam z ilosci
                     pamieci RAM serwera (patrz limit_km_z_pamieci), np. ok. 1,25 km przy 3,7 GiB
  FLOOD_MAX_JOBS     ile animacji naraz na calym serwerze (domyslnie 1)
  FLOOD_WCS_WORKERS  ile zapytan naraz do geoportalu (domyslnie 4)
"""
import hashlib
import hmac
import os
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import streamlit as st

INK, WATER, MIST, ALERT = "#12303F", "#1F7A8C", "#EAF2F5", "#E4572E"

# Model zuzycia pamieci (z jednego pomiaru, patrz potrzebna_pamiec_mb): baza + MB na km2 kadru (+ model U-Net)
PAMIEC_BAZA_MB, PAMIEC_NA_KM2_MB, PAMIEC_MODEL_MB = 300, 890, 500
UDZIAL_RAM_NA_KADR = 0.6      # jaka czesc calej pamieci wolno przeznaczyc na jedna animacje


def pamiec_calkowita_mb():
    """MemTotal z /proc/meminfo (Linux); None, gdy nie da sie ustalic."""
    try:
        with open("/proc/meminfo") as f:
            for linia in f:
                if linia.startswith("MemTotal:"):
                    return int(linia.split()[1]) / 1024
    except OSError:
        pass
    return None


def limit_km_z_pamieci(krok=0.25, minimum=0.5, maksimum=3.0):
    """Najwiekszy bok kadru (km), ktory miesci sie w UDZIALE calej pamieci serwera, z modelem U-Net.
    Zaokraglony W DOL do kroku. Bez informacji o pamieci zwraca ostrozne 1.0."""
    total = pamiec_calkowita_mb()
    if total is None:
        return 1.0
    budzet = UDZIAL_RAM_NA_KADR * total - PAMIEC_BAZA_MB - PAMIEC_MODEL_MB
    km = (max(budzet, 0) / PAMIEC_NA_KM2_MB) ** 0.5
    return min(maksimum, max(minimum, int(km / krok + 1e-9) * krok))


_ENV_KM = os.environ.get("FLOOD_MAX_KM")
MAX_KM = float(_ENV_KM) if _ENV_KM else limit_km_z_pamieci()
MAX_KM_AUTO = not _ENV_KM
MAX_JOBS = int(os.environ.get("FLOOD_MAX_JOBS", "1"))
WCS_WORKERS = int(os.environ.get("FLOOD_WCS_WORKERS", "4"))
ROOT = Path(tempfile.gettempdir()) / "flood_uploads"
WCS_CACHE = Path(tempfile.gettempdir()) / "flood_wcs_cache"
TTL_S = 2 * 3600

_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;700;800&display=swap');
html, body, [class*="css"], .stApp {{ font-family: 'Manrope', sans-serif; }}
.stApp {{ background: #F6F9FA; }}
.block-container {{ padding-top: 2rem; max-width: 1280px; }}
.stApp h1, .stApp h2, .stApp h3 {{ color: {INK}; }}
.stApp p, .stApp label, .stApp li {{ color: {INK}; }}
div[data-testid="stMetric"] {{ background: white; border: 1px solid #D5E3E9;
    border-radius: 12px; padding: .7rem 1rem; }}
div[data-testid="stMetricLabel"] p {{ color: #4A6572 !important; }}
div[data-testid="stMetricValue"] {{ color: {INK} !important; font-weight: 800; }}
</style>
"""


def styl():
    st.markdown(_CSS, unsafe_allow_html=True)




# ----------------------------------------------------------------------
# Logowanie (bramka na poczatku kazdej strony)
# ----------------------------------------------------------------------
PBKDF2_ITER = 200_000
MAX_BLEDOW = 10          # tyle nieudanych prob w oknie czasu blokuje logowanie dla wszystkich
OKNO_BLEDOW_S = 300


def hasz_hasla(haslo, sol_hex, iteracje=PBKDF2_ITER):
    """PBKDF2-HMAC-SHA256; zwraca hex. Haslo NIGDY nie jest zapisane jawnym tekstem."""
    return hashlib.pbkdf2_hmac("sha256", haslo.encode("utf-8"), bytes.fromhex(sol_hex), iteracje).hex()


def _konfiguracja_logowania():
    """(login, sol, hasz, iteracje) ze zmiennych srodowiskowych albo z .streamlit/secrets.toml; None = brak."""
    login, sol, hasz = (os.environ.get("FLOOD_LOGIN"), os.environ.get("FLOOD_SALT"),
                        os.environ.get("FLOOD_HASH"))
    iteracje = int(os.environ.get("FLOOD_ITER", PBKDF2_ITER))
    if not (login and sol and hasz):
        try:
            a = st.secrets["auth"]
            login, sol, hasz = a["login"], a["salt"], a["hash"]
            iteracje = int(a.get("iterations", PBKDF2_ITER))
        except Exception:
            return None
    return login, sol, hasz, iteracje


@st.cache_resource
def _nieudane_proby():
    return {"lock": threading.Lock(), "czasy": []}


def wymagaj_logowania():
    """
    Wolac na poczatku KAZDEJ strony (po st.set_page_config). Bez zalogowania pokazuje formularz
    i konczy skrypt (st.stop). Brak konfiguracji = dostep zablokowany (fail closed).
    Wylaczenie tylko jawnie: FLOOD_AUTH=off.
    """
    if os.environ.get("FLOOD_AUTH", "on").lower() == "off":
        return
    if st.session_state.get("zalogowany"):
        return

    cfg = _konfiguracja_logowania()
    _, srodek, _ = st.columns([1, 1.2, 1])
    with srodek:
        st.title("🔒 Logowanie")
        if cfg is None:
            st.error("Logowanie nie jest skonfigurowane na serwerze (brak sekcji [auth] w "
                     ".streamlit/secrets.toml), więc dostęp jest zablokowany.")
            st.stop()
        login_ok, sol, hasz_ok, iteracje = cfg
        with st.form("logowanie"):
            login = st.text_input("Login")
            haslo = st.text_input("Hasło", type="password")
            wyslano = st.form_submit_button("Zaloguj", type="primary", use_container_width=True)
        if wyslano:
            stan = _nieudane_proby()
            with stan["lock"]:
                teraz = time.time()
                stan["czasy"] = [t for t in stan["czasy"] if teraz - t < OKNO_BLEDOW_S]
                zablokowane = len(stan["czasy"]) >= MAX_BLEDOW
            if zablokowane:
                st.error("Zbyt wiele nieudanych prób logowania. Spróbuj ponownie za kilka minut.")
            else:
                # oba porownania zawsze wykonywane, stalym czasem
                ok_login = hmac.compare_digest(login.encode("utf-8"), login_ok.encode("utf-8"))
                ok_haslo = hmac.compare_digest(hasz_hasla(haslo, sol, iteracje), hasz_ok)
                if ok_login and ok_haslo:
                    st.session_state["zalogowany"] = True
                    st.rerun()
                with stan["lock"]:
                    stan["czasy"].append(time.time())
                time.sleep(1.5)                      # spowalnia zgadywanie
                st.error("Nieprawidłowy login lub hasło.")
    st.stop()


def wyloguj():
    st.session_state["zalogowany"] = False
    st.rerun()


# ----------------------------------------------------------------------
# Nawigacja: duzy pasek na gorze kazdej strony (zamiast malego menu bocznego)
# ----------------------------------------------------------------------
STRONY = [
    ("app.py", "Analiza miast", "🗺️"),
    ("pages/2_Wybierz_miejsce.py", "Symulacja dla miejsca", "📍"),
    ("pages/1_Wlasny_plik_tif.py", "Symulacja z pliku .tif", "📂"),
]

_NAV_CSS = f"""
<style>
/* domyslne menu boczne i pasek Streamlita chowamy - zastepuje je wlasny pasek na gorze */
section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"], [data-testid="stToolbar"], [data-testid="stDecoration"],
.stAppDeployButton, #MainMenu {{ display: none !important; }}
header[data-testid="stHeader"] {{ height: 0; min-height: 0; background: transparent; }}
.stApp .block-container {{ padding-top: 1.1rem; max-width: 1280px; }}

/* ---- pasek nawigacji ---- */
.st-key-nawigacja {{ background: white; border: 1px solid #D5E3E9; border-radius: 16px;
    padding: .45rem .8rem; margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(18,48,63,.06); }}
.st-key-nawigacja [data-testid="stHorizontalBlock"] {{ align-items: center; gap: .4rem; }}
.marka {{ font-weight: 800; font-size: 1.08rem; color: {INK}; white-space: nowrap; letter-spacing: -.01em; }}
.marka span {{ color: {WATER}; }}
[data-testid="stPageLink"] a, a[data-testid="stPageLink-NavLink"] {{
    justify-content: center; padding: .5rem .9rem; border-radius: 10px; border: 1px solid transparent;
    background: transparent; text-decoration: none; }}
[data-testid="stPageLink"] a p, a[data-testid="stPageLink-NavLink"] p {{
    font-size: .97rem !important; font-weight: 700 !important; color: #4A6572 !important; }}
[data-testid="stPageLink"] a:hover, a[data-testid="stPageLink-NavLink"]:hover {{ background: {MIST}; }}
[data-testid="stPageLink"] a[aria-current="page"], a[data-testid="stPageLink-NavLink"][aria-current="page"] {{
    background: {WATER}; }}
[data-testid="stPageLink"] a[aria-current="page"] p,
a[data-testid="stPageLink-NavLink"][aria-current="page"] p {{ color: white !important; }}
.st-key-nawigacja button {{ border: 1px solid #D5E3E9 !important; background: white !important; }}
.st-key-nawigacja button p {{ color: #4A6572 !important; font-weight: 700; font-size: .9rem; }}

/* ---- naglowek strony i kroki ---- */
.ph {{ display: flex; gap: 1rem; align-items: center; margin: .2rem 0 1rem 0; }}
.ph-ico {{ flex: none; width: 3.4rem; height: 3.4rem; border-radius: 14px; background: {MIST};
    display: flex; align-items: center; justify-content: center; font-size: 1.7rem; }}
.ph h1 {{ font-size: 2rem; font-weight: 800; letter-spacing: -.02em; margin: 0; padding: 0; line-height: 1.15; }}
.ph p {{ margin: .25rem 0 0 0; color: #4A6572; max-width: 50rem; font-size: 1rem; }}
.stp {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: 0 0 .6rem 0; }}
.stp .s {{ display: flex; align-items: center; gap: .5rem; padding: .3rem .8rem .3rem .35rem; border-radius: 999px;
    background: white; border: 1px solid #D5E3E9; color: #7A909B; font-weight: 700; font-size: .9rem; }}
.stp .s b {{ display: inline-flex; width: 1.6rem; height: 1.6rem; border-radius: 50%; align-items: center;
    justify-content: center; background: {MIST}; color: #7A909B; font-size: .85rem; }}
.stp .s.ok {{ color: {INK}; }}
.stp .s.ok b {{ background: {WATER}; color: white; }}
.krok {{ display: flex; gap: .8rem; align-items: center; margin: 1.9rem 0 .8rem 0; }}
.krok .nr {{ flex: none; width: 2.1rem; height: 2.1rem; border-radius: 50%; background: {WATER}; color: white;
    font-weight: 800; display: flex; align-items: center; justify-content: center; }}
.krok h3 {{ margin: 0; padding: 0; font-size: 1.3rem; font-weight: 800; line-height: 1.2; }}
.krok p {{ margin: .1rem 0 0 0; color: #4A6572; font-size: .93rem; }}
.chipy {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: .2rem 0 .4rem 0; }}
.chip {{ background: {MIST}; border-radius: 999px; padding: .3rem .85rem; font-size: .88rem; color: #4A6572; }}
.chip b {{ color: {INK}; }}

/* ---- karty i pola ---- */
[class*="st-key-karta_"] {{ background: white; border: 1px solid #D5E3E9 !important; border-radius: 16px;
    padding: 1.1rem 1.2rem; box-shadow: 0 1px 4px rgba(18,48,63,.05); }}
[data-testid="stExpander"] {{ border: 1px solid #D5E3E9; border-radius: 12px; background: white; }}
[data-testid="stFileUploaderDropzone"] {{ border: 2px dashed #9CC3CF; background: #F3F9FB; border-radius: 14px; }}
.stApp [data-testid="stBaseButton-primary"], .stApp button[kind="primary"] {{
    background: {WATER} !important; border: 1px solid {WATER} !important; border-radius: 12px;
    padding: .7rem 1.4rem; }}
.stApp [data-testid="stBaseButton-primary"] p, .stApp button[kind="primary"] p {{
    color: white !important; font-weight: 800; font-size: 1.05rem; }}
.stApp [data-testid="stBaseButton-primary"]:hover {{ background: #17606E !important; }}
.stTabs [data-baseweb="tab-list"] {{ gap: .4rem; }}
.stTabs button[data-baseweb="tab"] {{ height: 2.6rem; padding: 0 1.1rem; }}
.stTabs button[data-baseweb="tab"] p {{ color: #4A6572 !important; font-weight: 700; }}
.stTabs button[data-baseweb="tab"][aria-selected="true"] p {{ color: {WATER} !important; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {WATER} !important; }}
[data-testid="stTab"] p {{ color: #4A6572 !important; font-weight: 700; }}
[data-testid="stTab"][data-selected="true"] p {{ color: {WATER} !important; }}
[data-testid="stTab"] .react-aria-SelectionIndicator {{ background-color: {WATER} !important; }}
</style>
"""


def _css_aktywnej_strony(plik):
    """Nowsze wersje Streamlita nie oznaczaja aktywnego linku atrybutem aria-current, wiec podswietlamy go
    po adresie: pages/1_Wlasny_plik_tif.py -> href konczacy sie na 'Wlasny_plik_tif', app.py -> pusty href."""
    import re
    nazwa = Path(plik).stem
    if nazwa == "app":
        sel = ", ".join(f'.st-key-nawigacja a[data-testid="stPageLink-NavLink"][href="{h}"]' for h in ("", "/", "./"))
    else:
        slug = re.sub(r"^\d+_", "", nazwa)
        sel = f'.st-key-nawigacja a[data-testid="stPageLink-NavLink"][href$="{slug}"]'
    sel_p = ", ".join(x.strip() + " p" for x in sel.split(","))
    return (f"<style>{sel} {{ background: {WATER} !important; }} "
            f"{sel_p} {{ color: white !important; }}</style>")


def naglowek(ikona, tytul, opis):
    """Naglowek strony: ikona, tytul i jedno zdanie opisu."""
    st.markdown(f'<div class="ph"><div class="ph-ico">{ikona}</div><div><h1>{tytul}</h1><p>{opis}</p></div></div>',
                unsafe_allow_html=True)


def postep(kroki, ile_gotowych):
    """Pasek krokow: pierwsze `ile_gotowych` krokow jest podswietlonych."""
    el = "".join(f'<div class="s {"ok" if i < ile_gotowych else ""}"><b>{i + 1}</b>{n}</div>'
                 for i, n in enumerate(kroki))
    return f'<div class="stp">{el}</div>'


def krok(nr, tytul, podpis=""):
    """Naglowek sekcji z numerem."""
    st.markdown(f'<div class="krok"><div class="nr">{nr}</div><div><h3>{tytul}</h3>'
                f'{f"<p>{podpis}</p>" if podpis else ""}</div></div>', unsafe_allow_html=True)


def karta(klucz):
    """Biala karta (kontener z ramka); klucz musi byc unikalny na stronie."""
    try:
        return st.container(border=True, key="karta_" + klucz)
    except TypeError:                       # starsza wersja Streamlita bez `key`
        return st.container(border=True)


def chipy(pozycje):
    """Wiersz malych etykiet [(nazwa, wartosc), ...] zamiast duzych kafli z metrykami."""
    st.markdown('<div class="chipy">' + "".join(f'<span class="chip">{n}: <b>{w}</b></span>'
                                                for n, w in pozycje) + "</div>", unsafe_allow_html=True)


def nawigacja():
    """Pasek nawigacji. Wolac PO st.set_page_config(), na poczatku kazdej strony (rowniez app.py)."""
    st.markdown(_NAV_CSS, unsafe_allow_html=True)
    try:
        import inspect
        st.markdown(_css_aktywnej_strony(inspect.stack()[1].filename), unsafe_allow_html=True)
    except Exception:
        pass                                 # brak podswietlenia aktywnej strony nie moze psuc strony
    try:
        kontener = st.container(key="nawigacja")
    except TypeError:                       # starsza wersja Streamlita bez `key`
        kontener = st.container()
    with kontener:
        z_wylogowaniem = bool(st.session_state.get("zalogowany"))
        marka, menu, wyj = st.columns([2.3, 6, 1.1] if z_wylogowaniem else [2.3, 6, 0.01])
        marka.markdown('<div class="marka">🌊 Analiza <span>powodziowa AI</span></div>', unsafe_allow_html=True)
        with menu:
            for kol, (sciezka, etykieta, ikona) in zip(st.columns(len(STRONY)), STRONY):
                with kol:
                    try:
                        st.page_link(sciezka, label=etykieta, icon=ikona, use_container_width=True)
                    except TypeError:       # wersja bez use_container_width
                        st.page_link(sciezka, label=etykieta, icon=ikona)
        if z_wylogowaniem:
            with wyj:
                if st.button("Wyloguj", key="wyloguj", use_container_width=True):
                    wyloguj()


# ----------------------------------------------------------------------
# Pliki tymczasowe
# ----------------------------------------------------------------------
def sprzataj():
    """Kasuje stare katalogi sesji (wgrane .tif i wygenerowane GIF-y)."""
    if not ROOT.exists():
        return
    teraz = time.time()
    for d in ROOT.iterdir():
        try:
            if teraz - d.stat().st_mtime > TTL_S:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


def katalog_sesji():
    sid = st.session_state.setdefault("sid", uuid.uuid4().hex[:12])
    d = ROOT / sid
    d.mkdir(parents=True, exist_ok=True)
    return d


# ----------------------------------------------------------------------
# Zasoby wspolne: semafor zadan, model
# ----------------------------------------------------------------------
@st.cache_resource
def sloty():
    return threading.BoundedSemaphore(MAX_JOBS)


def model_dostepny():
    return Path("best_model.pth").exists()


@st.cache_resource(show_spinner="Ładuję model U-Net…")
def zaladuj_model():
    import segmentation_models_pytorch as smp
    import torch
    device = torch.device("cpu")
    model = smp.Unet(encoder_name="resnet34", in_channels=1, classes=1).to(device)
    ckpt = torch.load("best_model.pth", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
    model.eval()
    return model, device


def funkcja_predykcji(model, device):
    import torch

    def f(arr):
        h, w = arr.shape
        ph, pw = (-h) % 32, (-w) % 32
        a = np.pad(arr, ((0, ph), (0, pw)), mode="reflect")
        t = torch.from_numpy(a.astype(np.float32))[None, None].to(device)
        with torch.no_grad():
            p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
        return p[:h, :w]
    return f


# ----------------------------------------------------------------------
# Zabezpieczenie pamieci
# ----------------------------------------------------------------------
def wolna_pamiec_mb():
    """MemAvailable z /proc/meminfo (Linux); None, gdy nie da sie ustalic."""
    try:
        with open("/proc/meminfo") as f:
            for linia in f:
                if linia.startswith("MemAvailable:"):
                    return int(linia.split()[1]) / 1024
    except OSError:
        pass
    return None


def potrzebna_pamiec_mb(km, z_modelem):
    """Szacunek z jednego pomiaru: ok. 1,2 GB dla kadru 1 km (2000x2000 px); rosnie z polem kadru.
    +500 MB na model U-Net to zalozenie (niezmierzone)."""
    return PAMIEC_BAZA_MB + PAMIEC_NA_KM2_MB * km ** 2 + (PAMIEC_MODEL_MB if z_modelem else 0)


def zarezerwuj(km, z_modelem):
    """Zwraca (semafor, blad). Gdy blad nie jest None, zadania nie wolno uruchamiac."""
    potrzeba = potrzebna_pamiec_mb(km, z_modelem)
    dostepna = wolna_pamiec_mb()
    if dostepna is not None and potrzeba > 0.85 * dostepna:
        return None, (f"Za mało wolnej pamięci na serwerze: ta animacja potrzebuje ok. {potrzeba:.0f} MB, "
                      f"a dostępne jest {dostepna:.0f} MB. Zmniejsz kadr albo spróbuj za chwilę.")
    sem = sloty()
    if not sem.acquire(blocking=False):
        return None, "Serwer generuje teraz inną animację. Spróbuj ponownie za kilka minut."
    return sem, None


# ----------------------------------------------------------------------
# Wspolne kawalki interfejsu
# ----------------------------------------------------------------------
def kontrolki_scenariusza():
    """Suwak i krok scenariusza. Zwraca (max_wzrost_m, krok_m) po przycieciu do 9 poziomow."""
    c1, c2 = st.columns(2)
    max_rise = c1.slider("Maksymalny wzrost ponad stan normalny (m)", 1.0, 8.0, 4.0, 0.5)
    krok = c2.radio("Krok scenariusza", [1.0, 0.5], horizontal=True,
                    format_func=lambda v: f"co {v:g} m")
    n_krokow = min(int(max_rise / krok + 1e-9), 9)
    max_eff = n_krokow * krok
    st.caption(f"Animacja pokaże stan normalny i {n_krokow} kolejnych poziomów, do +{max_eff:g} m. "
               "To wzrosty względem powierzchni rzeki widocznej w NMPT, a nie odczyty z wodowskazu.")
    return max_eff, krok


def pokaz_wynik(wynik, sig, nazwa_pliku="symulacja_powodzi.gif", dodatek=""):
    """Wyswietla gotowa animacje, jesli dotyczy biezacych ustawien (sig)."""
    if not (wynik and wynik["sig"] == sig and Path(wynik["path"]).exists()):
        return
    st.success(f"Gotowe w {wynik['t']:.0f} s.")
    for n in wynik["info"]["notes"]:
        st.warning(n)
    _, srodek, _ = st.columns([1, 4, 1])
    with srodek:
        st.image(wynik["path"], use_container_width=True)
    with open(wynik["path"], "rb") as f:
        st.download_button("Pobierz GIF", f.read(), file_name=nazwa_pliku, mime="image/gif")
    info = wynik["info"]
    st.caption(
        f"Powierzchnia rzeki w kadrze ≈ {info['surface']:.2f} m n.p.m. (stan normalny). "
        f"Wykryte budynki: {info['buildings_pct']:.1f}% powierzchni. Model gruntu: {info.get('ground_src', '-')}. "
        "Budynek uznajemy za zalany, gdy woda stoi przy jego ścianach; kolor oznacza średnią głębokość "
        "wody przy ścianach. " + dodatek
    )
    if info.get("curve"):
        import pandas as pd
        df = pd.DataFrame(info["curve"]).set_index("rise_m")[["area_pct", "bld_pct"]]
        df.columns = ["Zalany teren [%]", "Zalane budynki [%]"]
        with st.expander("Wykres: ile zalewa przy kolejnych poziomach wody"):
            st.line_chart(df)
            st.caption("Oś pozioma: wzrost wody ponad stan normalny [m]. Gwałtowny skok oznacza moment, "
                       "w którym woda przelewa się przez brzeg, nabrzeże lub wał i zalewa położony za nim teren.")