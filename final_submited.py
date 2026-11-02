import os
import time
import datetime

# --- 1. STRICT ISOLATION ---
os.environ["CUDA_VISIBLE_DEVICES"] = "3" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import glob
import numpy as np
import torch
import torch.nn as nn
import base64
import pandas as pd
from io import BytesIO
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from basicsr.archs.swinir_arch import SwinIR

# --- 2. DEVICE SETUP ---
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Targeting: {DEVICE} | Name: {torch.cuda.get_device_name(DEVICE)}")

# --- 3. HYPERPARAMETERS ---
BATCH_SIZE = 8
EPOCHS = 100  # Bumped to 100 for maximum Kaggle performance
LR = 2e-4

# --- 4. DATASETS ---
class NpyDataset(Dataset):
    def __init__(self, gt_dir, noisy_dir):
        self.gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.npy")))
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))
    def __len__(self): return len(self.gt_files)
    def __getitem__(self, idx):
        clean = np.load(self.gt_files[idx]).astype(np.float32)
        noisy = np.load(self.noisy_files[idx]).astype(np.float32)
        if clean.ndim == 2: clean, noisy = clean[None, ...], noisy[None, ...]
        return torch.from_numpy(noisy), torch.from_numpy(clean)

class TestDataset(Dataset):
    def __init__(self, noisy_dir):
        self.files = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        noisy = np.load(path).astype(np.float32)
        if noisy.ndim == 2: noisy = noisy[None, ...]
        return torch.from_numpy(noisy), os.path.basename(path)

# --- 5. METRICS ---
def get_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    return 20 * torch.log10(1.0 / torch.sqrt(mse)) if mse > 0 else torch.tensor(100.0)

# --- 6. MAIN PIPELINE ---
def run_pipeline():
    TRAIN_GT, TRAIN_NOISY, TEST_NOISY = 'train/GT', 'train/NoisyLR', 'test/NoisyLR'
    SUBMISSION_DIR, BEST_MODEL_PATH = 'submission', 'best_swinir_model.pth'
    os.makedirs(SUBMISSION_DIR, exist_ok=True)

    loader = DataLoader(NpyDataset(TRAIN_GT, TRAIN_NOISY), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    model = SwinIR(upscale=2, in_chans=1, img_size=128, window_size=8,
                   img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, 
                   num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle').to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    criterion = nn.L1Loss()
    scaler = GradScaler()
    
    best_psnr = -1
    global_start_time = time.time()

    print(f"\n--- Starting Training for {EPOCHS} Epochs ---")
    for epoch in range(EPOCHS):
        model.train()
        epoch_psnr = 0
        epoch_start_time = time.time()
        
        for batch_idx, (noisy, clean) in enumerate(loader):
            noisy, clean = noisy.to(DEVICE, non_blocking=True), clean.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            
            with autocast():
                out = model(noisy)
                loss = criterion(out, clean)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                epoch_psnr += get_psnr(out, clean).item()
                
            if batch_idx % 50 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.5f}")
                
        scheduler.step()
        avg_psnr = epoch_psnr / len(loader)
        
        # --- TIME & ETA CALCULATION ---
        epoch_duration = time.time() - epoch_start_time
        time_elapsed = time.time() - global_start_time
        avg_time_per_epoch = time_elapsed / (epoch + 1)
        time_remaining = avg_time_per_epoch * (EPOCHS - epoch - 1)
        
        eta_string = str(datetime.timedelta(seconds=int(time_remaining)))
        duration_string = str(datetime.timedelta(seconds=int(epoch_duration)))
        
        print(f">>> Epoch {epoch} Complete | PSNR: {avg_psnr:.2f} dB | Epoch Time: {duration_string} | ETA: {eta_string}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"*** New Best Model Saved! (PSNR: {best_psnr:.2f} dB) ***")

    # --- HACKATHON INFERENCE ---
    print("\n--- Generating Hackathon Submission ---")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    
    test_loader = DataLoader(TestDataset(TEST_NOISY), batch_size=1)
    rows = []
    
    with torch.no_grad():
        for noisy, fname in test_loader:
            pred = model(noisy.to(DEVICE)).cpu().numpy().squeeze()
            pred = np.clip(pred, 0, 1).astype(np.float32)
            np.save(os.path.join(SUBMISSION_DIR, fname[0]), pred)
            
            buf = BytesIO()
            np.save(buf, pred)
            rows.append({"id": len(rows) + 1, "npy_base64": base64.b64encode(buf.getvalue()).decode()})

    pd.DataFrame(rows).to_csv("submission.csv", index=False)
    print("Done! Check 'submission.csv' and the 'submission/' folder.")

if __name__ == "__main__":
    run_pipeline()