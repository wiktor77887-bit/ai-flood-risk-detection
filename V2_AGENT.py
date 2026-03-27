import rasterio
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt

# ==========================================
# 1. PRZYGOTOWANIE DANYCH 
# ==========================================
class NMTDataset(Dataset):
    def __init__(self, asc_file, tif_file, patch_size=256):
        # Wczytanie przyciętego modelu terenu (NMPT)
        with rasterio.open(asc_file) as src:
            self.nmt = src.read(1) 
        
        # Wczytanie przyciętej maski
        with rasterio.open(tif_file) as src:
            self.mask = src.read(1)

        # KULOODPORNA NAPRAWA BŁĘDU WYMIARÓW:
        real_h = min(self.nmt.shape[0], self.mask.shape[0])
        real_w = min(self.nmt.shape[1], self.mask.shape[1])
        
        self.nmt = self.nmt[:real_h, :real_w]
        self.mask = self.mask[:real_h, :real_w]

        self.patch_size = patch_size
        self.h, self.w = self.nmt.shape
        
        self.patches_y = self.h // patch_size
        self.patches_x = self.w // patch_size
        self.total_patches = self.patches_y * self.patches_x

    def __len__(self):
        return self.total_patches

    def __getitem__(self, idx):
        y_idx = idx // self.patches_x
        x_idx = idx % self.patches_x
        y_start = y_idx * self.patch_size
        x_start = x_idx * self.patch_size
        
        # Wycinka kafelka
        nmt_patch = self.nmt[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        mask_patch = self.mask[y_start:y_start+self.patch_size, x_start:x_start+self.patch_size]
        
        # ==========================================
        # INTELIGENTNA NORMALIZACJA KONTRASTU
        # ==========================================
        valid_pixels = nmt_patch[nmt_patch > 0]
        
        if len(valid_pixels) > 0:
            min_val = np.min(valid_pixels)
            max_val = np.max(valid_pixels)
            nmt_patch = np.clip(nmt_patch, min_val, max_val)
            if max_val - min_val > 0:
                nmt_patch = (nmt_patch - min_val) / (max_val - min_val)
        else:
            nmt_patch = np.zeros_like(nmt_patch)
        
        nmt_tensor = torch.tensor(nmt_patch, dtype=torch.float32).unsqueeze(0)
        mask_tensor = torch.tensor(mask_patch, dtype=torch.float32).unsqueeze(0)
        
        return nmt_tensor, mask_tensor

# ==========================================
# 2. KONFIGURACJA ŚCIEŻEK
# ==========================================
# TUTAJ PODMIENIONO PLIKI NA NOWE WYCINKI!
NMT_FILE = "nmpt_wycinek.tif" 
MASK_FILE = "maska_wycinek.tif" 

print("Wczytywanie i cięcie danych...")
dataset = NMTDataset(NMT_FILE, MASK_FILE)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(f"Wygenerowano {len(dataset)} kafelków treningowych!")

# ==========================================
# 3. BUDOWA SIECI U-NET I URZĄDZENIE
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Trenujemy na: {device.type.upper()}")

# Używamy wstępnie wytrenowanych wag (Transfer Learning)
model = smp.Unet(
    encoder_name="resnet18", 
    encoder_weights="imagenet", # <--- Daje sieci "wiedzę" o kształtach
    in_channels=1, 
    classes=1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

# ==========================================
# 4. PĘTLA TRENINGOWA
# ==========================================
print("\nRozpoczynamy ostateczny trening inżynierski AI...")
EPOCHS = 7  # <--- 100 epok, żeby AI nauczyło się ignorować drzewa

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_nmt, batch_mask in dataloader:
        batch_nmt = batch_nmt.to(device)
        batch_mask = batch_mask.to(device)
        
        optimizer.zero_grad()
        predictions = model(batch_nmt)
        loss = criterion(predictions, batch_mask)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoka {epoch+1}/{EPOCHS} | Średni błąd (Loss): {avg_loss:.4f}")

print("\nTrening zakończony pomyślnie!")

# ==========================================
# 4.5. ZAPISYWANIE WYTRENOWANEGO MODELU
# ==========================================
torch.save(model.state_dict(), "unet_budynki_final.pth")
print("Model został zapisany na dysku jako 'unet_budynki_final.pth'!")

# ==========================================
# 5. TESTOWANIE MODELU (Wizualizacja wyników)
# ==========================================
print("\nGenerowanie predykcji na żywo...")

model.eval()

# Bierzemy losową paczkę do testów
test_nmt, test_mask = next(iter(dataloader))
test_nmt = test_nmt.to(device)

with torch.no_grad():
    raw_prediction = model(test_nmt)
    prediction = torch.sigmoid(raw_prediction)
    prediction_binary = (prediction > 0.5).float()

# Pobieramy pierwszy kafel z paczki z powrotem na CPU
img_nmt = test_nmt[0].cpu().squeeze().numpy()
img_true_mask = test_mask[0].cpu().squeeze().numpy()
img_pred_mask = prediction_binary[0].cpu().squeeze().numpy()

# Rysowanie pięknego wykresu
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_nmt, cmap='viridis') # 'viridis' pięknie pokazuje wysokość NMPT
axes[0].set_title("Model Terenu NMPT (Wejście)")

axes[1].imshow(img_true_mask, cmap='gray')
axes[1].set_title("Prawdziwe Budynki z QGIS")

axes[2].imshow(img_pred_mask, cmap='gray')
axes[2].set_title("Predykcja Twojego AI")

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()