#!/usr/bin/env python3
"""
ustaw_haslo.py - ustawia login i haslo do aplikacji (zapisuje SKROT hasla, nie haslo).

Uzycie (na serwerze, w folderze aplikacji):
    python3 ustaw_haslo.py

Zapisuje .streamlit/secrets.toml i ustawia prawa 600 (tylko wlasciciel). Po zmianie zrestartuj Streamlit.
Wybierz haslo o dlugosci co najmniej 12 znakow (najlepiej fraza z kilku slow).
"""
import getpass
import hashlib
import os
import sys
from pathlib import Path

ITER = 200_000
login = input("Login [Admin]: ").strip() or "Admin"
p1 = getpass.getpass("Nowe hasło: ")
p2 = getpass.getpass("Powtórz hasło: ")
if p1 != p2:
    sys.exit("Hasła się różnią. Nic nie zapisano.")
if len(p1) < 12:
    print("UWAGA: hasło ma mniej niż 12 znaków - łatwiej je złamać, jeśli ktoś zdobędzie plik secrets.toml.")
    if input("Zapisać mimo to? [t/N]: ").strip().lower() != "t":
        sys.exit("Przerwano.")

sol = os.urandom(16).hex()
hasz = hashlib.pbkdf2_hmac("sha256", p1.encode("utf-8"), bytes.fromhex(sol), ITER).hex()
cel = Path(".streamlit/secrets.toml")
cel.parent.mkdir(exist_ok=True)
cel.write_text(
    "# Dane logowania. Hasło NIE jest zapisane jawnym tekstem (tylko skrót PBKDF2).\n"
    "[auth]\n"
    f'login = "{login}"\n'
    f'salt = "{sol}"\n'
    f'hash = "{hasz}"\n'
    f"iterations = {ITER}\n",
    encoding="utf-8",
)
os.chmod(cel, 0o600)
print(f"Zapisano {cel} (prawa 600). Zrestartuj Streamlit.")
