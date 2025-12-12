import os
import cv2
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler

# ============================================================
#   1. RESIDUAL MODEL ARCHITECTURE (The "Refiner")
# ============================================================

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        res = self.conv1(x)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res  # The "Residual" connection

class ChromaRefiner(nn.Module):
    def __init__(self, in_channels=3, features=64, num_blocks=8):
        super().__init__()
        
        # 1. Head: Extract features from Y + Blurry Chroma
        self.head = nn.Conv2d(in_channels, features, 3, padding=1)
        
        # 2. Body: Deep processing to find edge correlations
        layers = []
        for _ in range(num_blocks):
            layers.append(ResBlock(features))
        self.body = nn.Sequential(*layers)
        
        # 3. Tail: Predict the Correction (Delta)
        # We only output 2 channels: Delta_Cr and Delta_Cb
        self.tail = nn.Conv2d(features, 2, 3, padding=1)

    def forward(self, x):
        """
        Input x: [Batch, 3, H, W] -> (Y, Cr_bilinear, Cb_bilinear)
        Output : [Batch, 3, H, W] -> (Y, Cr_fixed,    Cb_fixed)
        """
        # Save the "Base" (Bilinear Chroma) to add later
        # x[:, 1:3] contains Cr and Cb
        base_chroma = x[:, 1:3, :, :]
        
        # Pass everything (including Y for structural guidance) into the net
        feat = self.head(x)
        feat = self.body(feat)
        
        # Predict the CORRECTION (Delta)
        delta = self.tail(feat)
        
        # Final Result = Base + Correction
        refined_chroma = base_chroma + delta
        
        # Reassemble with the original Y (Pass-through)
        return torch.cat([x[:, 0:1, :, :], refined_chroma], dim=1)

# ============================================================
#   2. DATASET (Robust YUV Loader)
# ============================================================

class YUVChromaDataset(Dataset):
    def __init__(self, file_list, crop):
        self.files = file_list
        self.crop = crop
        self.max_retry = 10

    def __len__(self):
        return len(self.files)

    def safe_load(self, path):
        try:
            img = cv2.imread(path)
            if img is None: return None
            # OpenCV loads BGR, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except: return None

    def __getitem__(self, idx):
        n = len(self.files)
        for _ in range(self.max_retry):
            path = self.files[idx]
            img = self.safe_load(path)
            if img is None:
                idx = (idx + 1) % n
                continue
            
            H, W = img.shape[:2]
            if H < self.crop or W < self.crop:
                idx = (idx + 1) % n
                continue
                
            # Random Crop
            y = random.randint(0, H - self.crop)
            x = random.randint(0, W - self.crop)
            crop = img[y:y+self.crop, x:x+self.crop]
            
            # RGB -> YCrCb (Float 0-1)
            yuv = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
            
            Y  = yuv[:, :, 0:1]
            Cr = yuv[:, :, 1:2]
            Cb = yuv[:, :, 2:3]
            
            # --- SIMULATE 4:2:0 ARTIFACTS ---
            # 1. Downsample (Destroy info)
            h_sub, w_sub = self.crop // 2, self.crop // 2
            Cr_lo = cv2.resize(Cr, (w_sub, h_sub), interpolation=cv2.INTER_AREA)
            Cb_lo = cv2.resize(Cb, (w_sub, h_sub), interpolation=cv2.INTER_AREA)
            
            # 2. Upsample (Bilinear - This creates the "Base" prediction)
            Cr_input = cv2.resize(Cr_lo, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)[..., None]
            Cb_input = cv2.resize(Cb_lo, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)[..., None]
            
            # Input: [Y, Cr_bilinear, Cb_bilinear]
            inp = np.concatenate([Y, Cr_input, Cb_input], axis=2)
            
            # Target: [Y, Cr_original, Cb_original]
            tgt = np.concatenate([Y, Cr, Cb], axis=2)
            
            inp_t = torch.from_numpy(inp).permute(2,0,1)
            tgt_t = torch.from_numpy(tgt).permute(2,0,1)
            
            return inp_t, tgt_t
            
        return None

def safe_collate(batch):
    batch = [b for b in batch if b is not None]
    if len(batch) == 0: return None
    inputs, targets = zip(*batch)
    return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)

# ============================================================
#   3. VISUALIZATION HELPER
# ============================================================

def save_sample(inp, tgt, pred, epoch, save_dir):
    """
    Saves a comparison image: Input (Bilinear) | Target (GT) | Prediction (AI)
    Input tensors are (C, H, W) in YUV range [0,1]
    """
    def tensor_to_bgr(t):
        # Detach and move to CPU
        img = t.detach().float().cpu().permute(1, 2, 0).numpy()
        img = np.clip(img, 0.0, 1.0)
        # YCrCb -> BGR (via RGB)
        rgb = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_YCrCb2RGB)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return (bgr * 255).astype(np.uint8)

    img_in = tensor_to_bgr(inp)
    img_tg = tensor_to_bgr(tgt)
    img_pd = tensor_to_bgr(pred)

    # Stack horizontally
    combined = np.hstack([img_in, img_tg, img_pd])
    
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, f"val_epoch_{epoch:03d}.png"), combined)

# ============================================================
#   4. TRAINING LOOP
# ============================================================

def train_one_epoch(model, loader, opt, scaler, device, epoch, samples_dir):
    model.train()
    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    losses = []
    
    # L1 Loss is standard for PSNR maximization
    criterion = nn.L1Loss()
    
    first_batch = True
    
    for batch in pbar:
        if batch is None: continue
        
        inp, tgt = batch
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        
        opt.zero_grad(set_to_none=True)
        
        with autocast():
            # Forward pass
            output = model(inp)
            
            # Calculate loss ONLY on Chroma channels (indices 1 and 2)
            loss = criterion(output[:, 1:, :, :], tgt[:, 1:, :, :])

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        
        losses.append(loss.item())
        pbar.set_postfix(L1_Loss=f"{loss.item():.6f}")

        # Save sample from first batch only
        if first_batch:
            save_sample(inp[0], tgt[0], output[0], epoch, samples_dir)
            first_batch = False

    return np.mean(losses)

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Folder containing training images")
    ap.add_argument("--batch", type=int, default=16, help="Batch size")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--out", default="model_upscailing_v6") 
    ap.add_argument("--samples", default="samples_v6", help="Folder to save validation images")
    args = ap.parse_args()

    # Setup
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.samples, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device} | Output: {args.out} | Samples: {args.samples}")

    # Data
    print("Scanning files...")
    all_files = sorted(str(p) for p in Path(args.src).rglob("*") if p.suffix.lower() in [".jpg", ".png", ".webp"])
    
    # 97% Train / 3% Validation Split
    split = int(len(all_files) * 0.97)
    train_files = all_files[:split]
    
    train_ds = YUVChromaDataset(train_files, crop=args.crop)
    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=6, pin_memory=True, collate_fn=safe_collate)

    # Model
    print("Building Residual Model...")
    model = ChromaRefiner(features=64, num_blocks=8).to(device)
    
    # Optimizer
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Loop
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_ld, opt, scaler, device, epoch, args.samples)
        print(f"Epoch {epoch} | Chroma L1 Loss: {loss:.6f}")
        
        # Save Checkpoint
        state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save({"model": state_dict, "epoch": epoch}, os.path.join(args.out, f"res_epoch_{epoch:03d}.pth"))

    print("✅ Training complete.")

if __name__ == "__main__":
    main()