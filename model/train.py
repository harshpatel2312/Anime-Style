import os
import time
import torch
from torch import nn
from torch.utils.data import DataLoader
from itertools import cycle

from dataloader import get_dataloader
from generator import Generator
from discriminator import Discriminator
from losses import GANLoss

# ========= CONFIG ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 2
lr = 2e-4
log_interval = 10

# Paths
data_root = "/content/drive/MyDrive/Colab Notebooks/Anime-Style/data"
anime_train = f"{data_root}/anime/train"
real_train = f"{data_root}/real/train"

# ========= DATA ============
anime_train_loader, real_train_loader = get_dataloader(anime_train, real_train)

# ========= MODELS ============
G = Generator().to(device)
D = Discriminator().to(device)

# ========= LOSSES ============
criterion_gan = GANLoss(use_mse=True).to(device)
criterion_l1 = nn.L1Loss().to(device)

# ========= OPTIMIZERS ============
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

# ========= RESUME ============
resume_epoch = 0
resume_batch = 0

# Check if there is an existing checkpoint folder to auto-resume:
# Example: checkpoints/epoch_1_batch_400/checkpoint.pth
checkpoint_dir = "checkpoints"
latest_checkpoint = None
if os.path.exists(checkpoint_dir):
    subfolders = sorted(os.listdir(checkpoint_dir))
    if subfolders:
        latest = subfolders[-1]  # assume folders named epoch_X_batch_Y
        checkpoint_path = os.path.join(checkpoint_dir, latest, "checkpoint.pth")
        if os.path.exists(checkpoint_path):
            print(f"Found latest checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            G.load_state_dict(checkpoint['G_state_dict'])
            D.load_state_dict(checkpoint['D_state_dict'])
            optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
            optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
            resume_epoch = checkpoint['epoch']
            resume_batch = checkpoint['batch']
            print(f"Resumed from Epoch {resume_epoch}, Batch {resume_batch}")
        else:
            print("No valid checkpoint file found.")
else:
    print("No checkpoint directory found — starting fresh.")

# ========= TRAINING ============
try:
    for epoch in range(resume_epoch, epochs + 1):
        G.train()
        D.train()
        epoch_loss_G = 0.0
        epoch_loss_D = 0.0
        start_time = time.time()

        for i, (anime_imgs, real_imgs) in enumerate(zip(anime_train_loader, cycle(real_train_loader))):

            if epoch == resume_epoch and i < resume_batch:
                continue  # Skip already done batches

            anime_imgs = anime_imgs.to(device)
            real_imgs = real_imgs.to(device)

            #### Train D ####
            optimizer_D.zero_grad()

            fake_anime = G(real_imgs).detach()
            pred_real = D(anime_imgs)
            pred_fake = D(fake_anime)

            loss_D_real = criterion_gan(pred_real, True)
            loss_D_fake = criterion_gan(pred_fake, False)
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            #### Train G ####
            optimizer_G.zero_grad()

            fake_anime = G(real_imgs)
            pred_fake_for_G = D(fake_anime)
            loss_G_GAN = criterion_gan(pred_fake_for_G, True)
            loss_G_L1 = criterion_l1(fake_anime, anime_imgs)
            loss_G = loss_G_GAN + 10.0 * loss_G_L1
            loss_G.backward()
            optimizer_G.step()

            epoch_loss_G += loss_G.item()
            epoch_loss_D += loss_D.item()

            if (i + 1) % log_interval == 0:
                print(f"[Epoch {epoch}/{epochs}] [Batch {i + 1}] "
                      f"Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

            #### SAVE CHECKPOINT EVERY 100 BATCHES ####
            if (i + 1) % 100 == 0:
                save_dir = f"checkpoints/epoch_{epoch}_batch_{i + 1}"
                os.makedirs(save_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'batch': i + 1,
                    'G_state_dict': G.state_dict(),
                    'D_state_dict': D.state_dict(),
                    'optimizer_G_state_dict': optimizer_G.state_dict(),
                    'optimizer_D_state_dict': optimizer_D.state_dict()
                }, os.path.join(save_dir, "checkpoint.pth"))
                print(f"Saved checkpoint at Epoch {epoch}, Batch {i + 1}")

        num_batches = len(anime_train_loader)
        print(f"→ Epoch {epoch} Done | Time: {time.time() - start_time:.1f}s | "
              f"Avg Loss_D: {epoch_loss_D / num_batches:.4f} | "
              f"Avg Loss_G: {epoch_loss_G / num_batches:.4f}")

except KeyboardInterrupt:
    print("Training interrupted — saving final checkpoint...")
    save_dir = f"checkpoints/epoch_{epoch}_batch_{i + 1}_INTERRUPTED"
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'batch': i + 1,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict()
    }, os.path.join(save_dir, "checkpoint.pth"))
    print(f"Final checkpoint saved at: {save_dir}")