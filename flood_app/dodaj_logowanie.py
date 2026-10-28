#!/usr/bin/env python3
"""
dodaj_logowanie.py - wpisuje do app.py wywolanie logowania i paska nawigacji (po st.set_page_config).
Dziala niezaleznie od tego, czy dodales juz sam pasek nawigacji. Robi kopie app.py.bak.

    python3 dodaj_logowanie.py
"""
import shutil
import sys

BLOK = "from ui_common import wymagaj_logowania, nawigacja\nwymagaj_logowania()\nnawigacja()\n"
STARY = "from ui_common import nawigacja\nnawigacja()\n"

p = "app.py"
s = open(p, encoding="utf-8").read()
if "wymagaj_logowania" in s:
    sys.exit("Logowanie jest już dodane, nic nie zmieniam.")

if STARY in s:
    nowy = s.replace(STARY, BLOK, 1)
else:
    i = s.find("st.set_page_config(")
    if i < 0:
        sys.exit("Nie znalazłem st.set_page_config( w app.py - wpisz ręcznie po nim:\n" + BLOK)
    depth, quote, k = 0, None, i + len("st.set_page_config")
    while k < len(s):
        c = s[k]
        if quote:
            if c == quote and s[k - 1] != "\\":
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                break
        k += 1
    end = s.index("\n", k) + 1
    nowy = s[:end] + "\n" + BLOK + s[end:]

shutil.copy(p, p + ".bak")
open(p, "w", encoding="utf-8").write(nowy)
print("Gotowe: app.py chroniony logowaniem. Kopia zapasowa: app.py.bak")
