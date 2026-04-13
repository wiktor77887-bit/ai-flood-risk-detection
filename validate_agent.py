import torch
import rasterio
import numpy as np
import segmentation_models_pytorch as smp

def run_validation(nmpt_path, mask_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧐 Egzamin Agenta na urządzeniu: {device}")

    # 1. Ładowanie modelu
    model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 2. Wczytanie danych
    with rasterio.open(nmpt_path) as src_nmt:
        nmt = src_nmt.read(1)
        meta = src_nmt.meta.copy()
    
    with rasterio.open(mask_path) as src_mask:
        ground_truth = src_mask.read(1)

    # Dopasowanie wymiarów (na wypadek różnic o kilka pikseli)
    h, w = min(nmt.shape[0], ground_truth.shape[0]), min(nmt.shape[1], ground_truth.shape[1])
    nmt, ground_truth = nmt[:h, :w], ground_truth[:h, :w]

    # 3. Normalizacja (identyczna jak w treningu V2)
    valid = nmt[nmt > -100]
    p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
    nmt_norm = np.clip((nmt - p2) / (p98 - p2 + 1e-6), 0, 1)
    
    input_tensor = torch.tensor(nmt_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # 4. Predykcja
    print("🧠 AI analizuje nowy teren...")
    with torch.no_grad():
        pred = torch.sigmoid(model(input_tensor))
        pred_mask = (pred > 0.5).cpu().numpy().astype('uint8')[0, 0]

    # 5. Obliczanie IoU (Intersection over Union)
    intersection = np.logical_and(pred_mask, ground_truth > 0.5).sum()
    union = np.logical_or(pred_mask, ground_truth > 0.5).sum()
    iou_score = (intersection / union) if union > 0 else 0

    print(f"\n📊 WYNIK EGZAMINU:")
    print(f"✅ IoU na nowym terenie: {iou_score:.2%}")

    # 6. Zapisanie wyniku do TIF
    meta.update(dtype=rasterio.uint8, count=1, nodata=0, height=h, width=w)
    with rasterio.open("WYNIK_TEST_NOWY.tif", 'w', **meta) as dst:
        dst.write(pred_mask, 1)
    
    print(f"💾 Mapa wynikowa zapisana jako: WYNIK_TEST_NOWY.tif")

if __name__ == "__main__":
    run_validation("Nowy_Teren.tif", "nowa_maska.tif", "unet_nmpt_model.pth")