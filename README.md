# 🌊 AI Flood Risk & Building Segmentation on NMPT Data

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-U--Net-EE4C2C.svg)](https://pytorch.org/)
[![GIS](https://img.shields.io/badge/GIS-Rasterio%20%7C%20GeoPandas-4E9A06.svg)]()

## 📌 O projekcie
Projekt badawczo-rozwojowy (PoC) mający na celu automatyczną detekcję i segmentację budynków na podstawie Numerycznych Modeli Pokrycia Terenu (NMPT) z polskiego Geoportalu. Narzędzie to stanowi kluczowy element wstępny w systemach predykcji ryzyka powodziowego (obliczanie powierzchni nieprzepuszczalnych wokół infrastruktury).

Projekt łączy inżynierię danych przestrzennych (GIS) z uczeniem maszynowym (Computer Vision).

## 🚀 Technologie
* **Machine Learning:** PyTorch, Segmentation Models (U-Net, ResNet18)
* **Data Engineering (GIS):** Rasterio, GeoPandas, Shapely, QGIS
* **Przetwarzanie danych:** NumPy, Matplotlib

## 🛠️ Kluczowe problemy inżynierskie rozwiązane w projekcie

1. **Wyzwanie: "Brudne dane" (Dirty Data) i uszkodzone wektory**
   * **Rozwiązanie:** Opracowanie skryptu pre-processingu (`przygotuj_dane.py`), który automatycznie filtruje puste geometrie i obiekty o nieskończonych współrzędnych (`inf`), a także zarządza konfliktami wartości `NoData` (np. `-9999` w Geoportalu vs rzutowanie na `uint8`).
2. **Wyzwanie: Precyzyjne łączenie danych rastrowych i wektorowych**
   * **Rozwiązanie:** Automatyczne wyliczanie *Bounding Box* z masek wektorowych i matematyczne przycinanie gigantycznych modeli `.asc` NMPT do mniejszych kafelków w ujednoliconym układzie współrzędnych **EPSG:2180 (PL-1992)**.
3. **Wyzwanie: Mylenie gładkich dachów z koronami drzew przez AI**
   * **Rozwiązanie:** Zastosowanie **Transfer Learningu**. Wykorzystanie enkodera `ResNet18` (wytrenowanego na zbiorze ImageNet) dostarczyło sieci U-Net gotową "wiedzę" o ekstrakcji krawędzi, co drastycznie zwiększyło skuteczność izolacji budynków od szumu środowiskowego.

## 📂 Struktura repozytorium
* `przygotuj_dane.py` - Pipeline do czyszczenia danych, ekstrakcji Bounding Box i generowania kafelków treningowych (.tif).
* `train.py` - Główny skrypt budujący sieć U-Net, normalizujący kontrast NMPT i przeprowadzający pętlę treningową.
* `unet_budynki_final.pth` - Wytrenowane wagi modelu (dostępne do pobrania w zakładce Releases).

## 📊 Wyniki działania modelu
Model przyjmuje surowy zrzut z lasera (NMPT) i z wysoką precyzją generuje binarną maskę budynków.

![Wyniki segmentacji U-Net](link_do_twojego_zdjecia.png)  
*(Wykres z lewej: Surowy NMPT, Środek: Ground Truth z QGIS, Prawa: Predykcja sieci U-Net)*

## 🔮 Dalsze plany rozwoju (Next Steps)
- [ ] Zwiększenie zbioru danych o kafelki z innych układów urbanistycznych (generalizacja).
- [ ] Wdrożenie Data Augmentation (rotacje, flipy, szum cyfrowy) w celu zapobiegania overfittingowi.
- [ ] Zmiana metryki oceny z klasycznego Loss na procentowe **IoU (Intersection over Union)**.
- [ ] Post-processing: Automatyczna wektoryzacja przewidzianych masek rastrowych z powrotem do wielokątów (.shp / .gpkg).

---
*Projekt stworzony w ramach poszerzania kompetencji z zakresu uczenia maszynowego i analizy danych geoprzestrzennych.*# ai-flood-risk-detection

