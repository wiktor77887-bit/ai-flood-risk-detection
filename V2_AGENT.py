import rasterio
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import os

# ==========================================
# 0. DIAGNOSTYKA SPRZĘTOWA
# ==========================================
print("--- DIAGNOSTYKA SPRZĘTOWA ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**2 
    print(f"✅ GPU: {torch.cuda.get_device_name(0)} ({vram:.0f} MB VRAM)")
else:
    print("❌ CPU Mode - Trening będzie bardzo wolny!")
print("------------------------------\n")

# ==========================================
# 1. DATASET Z AUGMENTACJĄ I ROBUST NORMALIZATION
# ==========================================
class NMTDataset(Dataset):
    def __init__(self, nmt_file, mask_file, patch_size=256):
        with rasterio.open(nmt_file) as src:
            self.nmt = src.read(1) 
        with rasterio.open(mask_file) as src:
            self.mask = src.read(1)

        # Dopasowanie wymiarów (na wypadek różnic o 1-2 piksele)
        h, w = min(self.nmt.shape[0], self.mask.shape[0]), min(self.nmt.shape[1], self.mask.shape[1])
        self.nmt, self.mask = self.nmt[:h, :w], self.mask[:h, :w]
        
        self.patch_size = patch_size
        self.patches_y, self.patches_x = h // patch_size, w // patch_size

    def __len__(self):
        return self.patches_y * self.patches_x

    def __getitem__(self, idx):
        y = (idx // self.patches_x) * self.patch_size
        x = (idx % self.patches_x) * self.patch_size
        
        nmt_p = self.nmt[y:y+self.patch_size, x:x+self.patch_size].copy()
        mask_p = self.mask[y:y+self.patch_size, x:x+self.patch_size].copy()
        
        # --- 1. NORMALIZACJA PERCENTYLOWA (Odporna na błędy krawędzi) ---
        valid = nmt_p[nmt_p > -100]
        if len(valid) > 0:
            p2, p98 = np.percentile(valid, 2), np.percentile(valid, 98)
            nmt_p = np.clip((nmt_p - p2) / (p98 - p2 + 1e-6), 0, 1)
        else:
            nmt_p = np.zeros_like(nmt_p)

        # --- 2. AUGMENTACJA (Losowe odbicia) ---
        # Uczy model, że budynek to budynek niezależnie od obrotu
        if random.random() > 0.5:
            nmt_p = np.flip(nmt_p, axis=0).copy()
            mask_p = np.flip(mask_p, axis=0).copy()
        if random.random() > 0.5:
            nmt_p = np.flip(nmt_p, axis=1).copy()
            mask_p = np.flip(mask_p, axis=1).copy()
            
        return torch.tensor(nmt_p, dtype=torch.float32).unsqueeze(0), \
               torch.tensor(mask_p, dtype=torch.float32).unsqueeze(0)

# ==========================================
# 2. INICJALIZACJA MODELU I FUNKCJI STRATY
# ==========================================
NMT_FILE = "NMPT_PROSTY_MASTER2.tif" 
MASK_FILE = "MASKA_FINALNA_AI.tif" 

dataset = NMTDataset(NMT_FILE, MASK_FILE)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Model U-Net z enkoderem ResNet18 (lekki i skuteczny)
model = smp.Unet(encoder_name="resnet18", in_channels=1, classes=1).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Hybrydowa funkcja straty: BCE (piksele) + Dice (kształt obiektów)
criterion_bce = nn.BCEWithLogitsLoss()
criterion_dice = smp.losses.DiceLoss(mode='binary')

def hybrid_loss(y_pred, y_true):
    return criterion_bce(y_pred, y_true) + criterion_dice(y_pred, y_true)

# Scheduler: Zmniejsza LR o połowę co 10 epok, by doprecyzować wagi
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# ==========================================
# 3. TRENING (Zwiększony do 30 epok dla lepszej jakości)
# ==========================================
EPOCHS = 100
print(f"🚀 Startujemy! Trening na {len(dataset)} kafelkach przez {EPOCHS} epok...")

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for b_nmt, b_mask in dataloader:
        b_nmt, b_mask = b_nmt.to(device), b_mask.to(device)
        
        optimizer.zero_grad()
        outputs = model(b_nmt)
        loss = hybrid_loss(outputs, b_mask)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoka {epoch+1:02d}/{EPOCHS} | Loss: {epoch_loss / len(dataloader):.4f} | LR: {current_lr:.6f}")

# Zapisujemy "mózg" agenta
torch.save(model.state_dict(), "unet_nmpt_model.pth")
print("\n✅ Model zapisany jako 'unet_nmpt_model.pth'")

# ==========================================
# 4. FINALNA WIZUALIZACJA TESTOWA
# ==========================================
model.eval()
with torch.no_grad():
    t_nmt, t_mask = next(iter(dataloader))
    pred = torch.sigmoid(model(t_nmt.to(device)))
    pred_binary = (pred > 0.5).float()

plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1); plt.imshow(t_nmt[0].squeeze(), cmap='viridis'); plt.title("Wejście NMPT")
plt.subplot(1, 3, 2); plt.imshow(t_mask[0].squeeze(), cmap='gray'); plt.title("Maska Urzędowa")
plt.subplot(1, 3, 3); plt.imshow(pred_binary[0].cpu().squeeze(), cmap='gray'); plt.title("Predykcja AI (Agent)")
plt.tight_layout()
plt.show()