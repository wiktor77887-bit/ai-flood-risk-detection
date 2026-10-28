import torch
import rasterio
import numpy as np
import segmentation_models_pytorch as smp
import sys

def run_inference(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Startuję predykcję na: {device}")
    
    # 1. Inicjalizacja modelu
    model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=1).to(device)
    model.load_state_dict(torch.load("unet_nmpt_model.pth", map_location=device))
    model.eval()

    # 2. Wczytanie danych
    print(f"📂 Wczytuję plik: {input_path}")
    with rasterio.open(input_path) as src:
        img = src.read(1)
        meta = src.meta.copy()
        nodata = src.nodata

    # 3. INTELIGENTNA NORMALIZACJA (Poprawka "syfu")
    # Ignorujemy wartości nodata i ekstremalne błędy na krawędziach
    mask_nodata = (img == nodata) | (img < -50) | (img > 1000)
    img_clean = np.where(mask_nodata, 0, img)

    # Używamy percentyli, żeby model nie głupiał przez błędy na brzegach
    valid_pixels = img[~mask_nodata]
    if valid_pixels.size > 0:
        p2 = np.percentile(valid_pixels, 2)
        p98 = np.percentile(valid_pixels, 98)
        img_norm = np.clip((img_clean - p2) / (p98 - p2 + 1e-6), 0, 1)
    else:
        img_norm = img_clean

    input_tensor = torch.tensor(img_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 4. Predykcja
    print("🧠 AI analizuje teren...")
    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor))
        mask = (pred > 0.5).cpu().numpy().astype('uint8')[0, 0]

    # CZYSZCZENIE "PLAMEK" (Post-processing w kodzie)
    from scipy import ndimage
    # Usuwamy małe, odizolowane grupy pikseli (szum)
    structure = np.ones((3, 3)) # definiuje sąsiedztwo
    labeled, ncomponents = ndimage.label(mask, structure)
    sizes = ndimage.sum(mask, labeled, range(ncomponents + 1))
    mask_size = sizes < 50  # Próg: usuń obiekty mniejsze niż 50 pikseli
    remove_pixel = mask_size[labeled]
    mask[remove_pixel] = 0
    
    # Zerujemy maskę tam, gdzie oryginalnie nie było danych (NoData)
    mask[mask_nodata] = 0

    # 5. Przygotowanie metadanych i zapis
    meta.update(
        dtype=rasterio.uint8, 
        count=1, 
        nodata=0
    )
    
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mask.astype(rasterio.uint8), 1)
    
    print(f"✅ Sukces! Maska zapisana w: {output_path}")

# --- TO JEST KLUCZOWY MOMENT: URUCHOMIENIE ---
if __name__ == "__main__":
    # Sprawdzamy czy podano argumenty w terminalu
    if len(sys.argv) > 2:
        run_inference(sys.argv[1], sys.argv[2])
    else:
        # Domyślne nazwy, jeśli zapomnisz wpisać argumentów
        run_inference("NMPT_PROSTY_MASTER2.tif", "WYNIK_AI.tif")