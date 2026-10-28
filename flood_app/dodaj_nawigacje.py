import shutil
import sys

p = "app.py"
s = open(p, encoding="utf-8").read()
if "nawigacja" in s:
    sys.exit("Już dodane, nic nie zmieniam.")
i = s.find("st.set_page_config(")
if i < 0:
    sys.exit("Nie znalazłem st.set_page_config( w app.py - wpisz te dwie linie ręcznie.")

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

shutil.copy(p, p + ".bak")
s = s[:end] + "\nfrom ui_common import nawigacja\nnawigacja()\n" + s[end:]
open(p, "w", encoding="utf-8").write(s)
print("Gotowe: dopisano pasek nawigacji. Kopia zapasowa: app.py.bak")
