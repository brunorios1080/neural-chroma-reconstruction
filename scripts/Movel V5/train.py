import os
import cv2
import math
import time
import random
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
#                   SAFE DATASET V5.1.2
# ============================================================
# Dataset that loads images, crops, converts to YUV, downsamples chroma,
# and returns input (Y + low-res CrCb) and target (Y + full-res CrCb).
# Implements robust error handling and logging of bad images.
# This simulates compression on the chroma channels only, bypasses need of 4K 4:4:4 images.

class YUVChromaDataset(Dataset):
    def __init__(self, file_list, crop):
        self.files = file_list
        self.crop = crop
        self.max_retry = 10
        self.bad_log = "bad_images_v5.log"

        with open(self.bad_log, "w") as f:
            f.write("=== Bad Image Log (V5.1.2) ===\n")

    def __len__(self):
        return len(self.files)

    def log_bad(self, path, reason):
        try:
            with open(self.bad_log, "a") as f:
                f.write(f"{path} --> {reason}\n")
        except Exception:
            # If logging fails, don't crash training
            pass

    def safe_load(self, path):
        """Load RGB image or return None."""
        try:
            img = cv2.imread(path)
            if img is None:
                self.log_bad(path, "cv2 load returned None")
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if len(img.shape) != 3 or img.shape[2] != 3:
                self.log_bad(path, f"Invalid shape {img.shape}")
                return None
            return img
        except Exception as e:
            self.log_bad(path, f"Load exception: {str(e)}")
            return None

    def random_crop(self, img, path):
        """Random crop to (crop x crop), or None if too small."""
        H, W = img.shape[:2]
        if H < self.crop or W < self.crop:
            self.log_bad(path, f"Too small for crop {self.crop}x{self.crop}: {H}x{W}")
            return None

        y = random.randint(0, H - self.crop)
        x = random.randint(0, W - self.crop)
        return img[y:y+self.crop, x:x+self.crop]

    def __getitem__(self, idx):
        """
        Try up to max_retry different images (cycling forward in the list).
        If all fail, return None – collate_fn will drop it.
        """
        n = len(self.files)
        start_idx = idx

        for attempt in range(self.max_retry):
            path = self.files[idx]

            img = self.safe_load(path)
            if img is None:
                idx = (idx + 1) % n
                continue

            crop = self.random_crop(img, path)
            if crop is None:
                idx = (idx + 1) % n
                continue

            try:
                # RGB -> YCrCb (OpenCV uses YCrCb order)
                yuv = cv2.cvtColor(crop, cv2.COLOR_RGB2YCrCb).astype(np.float32) / 255.0
            except Exception:
                self.log_bad(path, "YUV conversion failed")
                idx = (idx + 1) % n
                continue

            # Separate channels
            Y  = yuv[:, :, 0:1]  # (H,W,1)
            Cr = yuv[:, :, 1:2]
            Cb = yuv[:, :, 2:3]

            # Downsample chroma then upsample back (simulate 4:2:0)
            try:
                Hc, Wc = self.crop // 2, self.crop // 2

                Cr_lo = cv2.resize(Cr, (Wc, Hc), interpolation=cv2.INTER_AREA)
                Cb_lo = cv2.resize(Cb, (Wc, Hc), interpolation=cv2.INTER_AREA)

                if Cr_lo.ndim == 2:
                    Cr_lo = Cr_lo[..., None]
                if Cb_lo.ndim == 2:
                    Cb_lo = Cb_lo[..., None]

                Cr_lo = cv2.resize(Cr_lo, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)
                Cb_lo = cv2.resize(Cb_lo, (self.crop, self.crop), interpolation=cv2.INTER_LINEAR)

                if Cr_lo.ndim == 2:
                    Cr_lo = Cr_lo[..., None]
                if Cb_lo.ndim == 2:
                    Cb_lo = Cb_lo[..., None]
                if Y.ndim == 2:
                    Y = Y[..., None]
                if Cr.ndim == 2:
                    Cr = Cr[..., None]
                if Cb.ndim == 2:
                    Cb = Cb[..., None]

            except Exception:
                self.log_bad(path, "Chroma resize error")
                idx = (idx + 1) % n
                continue

            # Final shape checks
            try:
                inp = np.concatenate([Y, Cr_lo, Cb_lo], axis=2)   # (H,W,3)
                tgt = np.concatenate([Y, Cr,    Cb   ], axis=2)   # (H,W,3)

                if inp.shape != (self.crop, self.crop, 3) or tgt.shape != (self.crop, self.crop, 3):
                    self.log_bad(path, f"Unexpected shape inp={inp.shape}, tgt={tgt.shape}")
                    idx = (idx + 1) % n
                    continue

                inp_t = torch.from_numpy(inp).permute(2,0,1)  # (3,H,W)
                tgt_t = torch.from_numpy(tgt).permute(2,0,1)

                return inp_t, tgt_t

            except Exception:
                self.log_bad(path, "Tensor assembly error")
                idx = (idx + 1) % n
                continue

        # If we reach here, this index region is cursed – signal failure
        return None


# ============================================================
#           CUSTOM COLLATE: DROP FAILED SAMPLES
# ============================================================

def safe_collate(batch):
    """
    batch: list of (inp, tgt) or None
    Returns (inputs, targets) or None if no valid samples.
    """
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return None
    inputs, targets = zip(*batch)
    return torch.stack(inputs, dim=0), torch.stack(targets, dim=0)


# ============================================================
#               SIMPLE  (UNET-LITE)
# ============================================================
# U-Net generator with fewer channels for faster training.
# Suitable for 256x256 inputs.

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.block(x)

class UNet_G(nn.Module):
    def __init__(self):
        super().__init__()

        self.e1 = ConvBlock(3, 32)
        self.p1 = nn.MaxPool2d(2)

        self.e2 = ConvBlock(32, 64)
        self.p2 = nn.MaxPool2d(2)

        self.e3 = ConvBlock(64, 128)
        self.p3 = nn.MaxPool2d(2)

        self.e4 = ConvBlock(128, 256)

        self.u3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.d3 = ConvBlock(256, 128)

        self.u2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.d2 = ConvBlock(128, 64)

        self.u1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.d1 = ConvBlock(64, 32)

        self.out = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.p1(e1))
        e3 = self.e3(self.p2(e2))
        b  = self.e4(self.p3(e3))

        d3 = self.u3(b)
        d3 = self.d3(torch.cat([d3, e3], dim=1))

        d2 = self.u2(d3)
        d2 = self.d2(torch.cat([d2, e2], dim=1))

        d1 = self.u1(d2)
        d1 = self.d1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))


# ============================================================
#                       DISCRIMINATOR
# ============================================================
# PatchGAN discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        ch = 32
        self.model = nn.Sequential(
            nn.Conv2d(3,  ch,   4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch, ch*2, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch*2, ch*4, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch*4, ch*8, 4, 2, 1), nn.LeakyReLU(0.2, True),
            nn.Conv2d(ch*8, 1,    4, 1, 0)
        )

    def forward(self, x):
        return self.model(x)


# ============================================================
#                   SAMPLE IMAGE SAVER
# ============================================================

def save_sample(inp, tgt, fake, out_dir, epoch):
    inp = inp.detach().cpu().permute(1,2,0).numpy()
    tgt = tgt.detach().cpu().permute(1,2,0).numpy()
    fake = fake.detach().cpu().permute(1,2,0).numpy()

    # Show Y-only on left (grayscale from Y)
    Y_only = inp[:, :, 0:1]
    Y_rgb  = np.repeat(Y_only, 3, axis=2)

    canvas = np.concatenate([Y_rgb, tgt, fake], axis=1)
    canvas = np.clip(canvas, 0.0, 1.0)

    os.makedirs(out_dir, exist_ok=True)
    cv2.imwrite(
        os.path.join(out_dir, f"epoch_{epoch:03d}.png"),
        (canvas * 255).astype(np.uint8)[:, :, ::-1]  # RGB->BGR for OpenCV
    )


# ============================================================
#                   TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(G, D, loader, g_opt, d_opt, device, epoch, samples_dir):
    G.train()
    D.train()

    pbar = tqdm(loader, total=len(loader), desc=f"Epoch {epoch}")
    g_losses, d_losses = [], []

    for batch in pbar:
        # custom collate_fn may return None if whole batch was bad
        if batch is None:
            continue

        inp, tgt = batch
        inp = inp.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)

        # -----------------------------
        # Train Discriminator
        # -----------------------------
        d_opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            fake = G(inp)

        pred_real = D(tgt)
        pred_fake = D(fake.detach())

        loss_D = (
            F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real)) +
            F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        ) * 0.5

        loss_D.backward()
        d_opt.step()

        # -----------------------------
        # Train Generator
        # -----------------------------
        g_opt.zero_grad(set_to_none=True)

        fake = G(inp)
        pred_fake = D(fake)

        # GAN loss + L1 color reconstruction
        gan_loss = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
        l1_loss  = F.l1_loss(fake, tgt)
        loss_G   = gan_loss + 10.0 * l1_loss

        loss_G.backward()
        g_opt.step()

        g_losses.append(loss_G.item())
        d_losses.append(loss_D.item())

        pbar.set_postfix(G=f"{loss_G.item():.4f}", D=f"{loss_D.item():.4f}")

    # Save example from last batch if we had any
    if len(g_losses) > 0:
        save_sample(inp[0], tgt[0], fake[0], samples_dir, epoch)

    return (np.mean(g_losses) if g_losses else 0.0,
            np.mean(d_losses) if d_losses else 0.0)


# ============================================================
#                   MAIN TRAINING LOOP
# ============================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--out", default="checkpoints_v5_yuv_gan")
    ap.add_argument("--samples", default="samples_v5")
    args = ap.parse_args()

    all_files = sorted(
        str(p) for p in Path(args.src).rglob("*")
        if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
    )

    random.shuffle(all_files)
    split = int(len(all_files) * 0.97)
    train_files = all_files[:split]
    val_files   = all_files[split:]  # not used yet, but kept for later

    print(f"Train: {len(train_files)}  |  Val: {len(val_files)}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    train_ds = YUVChromaDataset(train_files, crop=args.crop)
    train_ld = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=safe_collate
    )

    G = UNet_G().to(device)
    D = Discriminator().to(device)

    if torch.cuda.device_count() > 1 and device == "cuda":
        print("Using DataParallel on", torch.cuda.device_count(), "GPUs")
        G = nn.DataParallel(G)
        D = nn.DataParallel(D)

    g_opt = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.samples, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        g_loss, d_loss = train_one_epoch(
            G, D, train_ld, g_opt, d_opt, device, epoch, args.samples
        )
        dt = (time.time() - t0) / 60.0

        print(f"Epoch {epoch}/{args.epochs} | G={g_loss:.4f}  D={d_loss:.4f}  | {dt:.2f} min")

        torch.save(
            {
                "G": G.state_dict(),
                "D": D.state_dict(),
                "g_loss": g_loss,
                "d_loss": d_loss,
                "epoch": epoch,
            },
            os.path.join(args.out, f"epoch_{epoch:03d}.pth")
        )

    print("✅ Training complete.")


if __name__ == "__main__":
    main()
