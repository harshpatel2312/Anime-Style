import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from generator import Generator
from discriminator import Discriminator
from losses import GANLoss
from dataloader import get_dataloader
from itertools import cycle
import time
import os

# ========== CONFIG ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 2
lr = 2e-4
log_interval = 10 # For debugging

# Paths to data folders
anime_train = "../data/anime/train/"
real_train = "../data/real/train/"

# ========== DATALOADERS ==========
anime_train_loader, real_train_loader = get_dataloader(anime_train, real_train)

# ========== MODELS ==========
G = Generator().to(device)
D = Discriminator().to(device)

# ========== LOSS FUNCTIONS ==========
criterion_gan = GANLoss(use_mse = True).to(device) # Adversarial Loss
criterion_l1 = nn.L1Loss().to(device)

# ========== OPTIMIZERS ==========
optimizer_G = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_D = torch.optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.999))

# ========== TRAINING ==========
for epoch in range(1, epochs + 1):
    # Set to training mode
    G.train() 
    D.train()

    epoch_loss_G = 0
    epoch_loss_D = 0
    start_time = time.time()

    for i, (real_imgs, anime_imgs) in enumerate(zip(cycle(real_train_loader), anime_train_loader)):
        if i >= len(anime_train_loader):
            break
            
        real_imgs = real_imgs.to(device)
        anime_imgs = anime_imgs.to(device)

        # ========================================
        # 1. Train Discriminator
        # ========================================
        optimizer_G.zero_grad()

        fake_anime = G(real_imgs).detach() # Exclude gradients
        pred_real = D(anime_imgs) # Predict real
        pred_fake = D(fake_anime) # Predict fake

        loss_D_real = criterion_gan(pred_real, True) # Calculate adversarial loss 
        loss_D_fake = criterion_gan(pred_fake, False) # Calculate recnstruction loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5 # Total loss
        loss_D.backward() # Backpropogation
        optimizer_D.step() # Update weights

        # ========================================
        # 2. Train Generator
        # ========================================
        optimizer_G.zero_grad()

        fake_anime = G(real_imgs) # Convert real -> anime
        pred_fake_for_G = D(fake_anime) # Step to fool dicriminator in identifying fake as real...
        loss_G_GAN = criterion_gan(pred_fake_for_G, True) # Calculate adversarial loss 
        loss_G_L1 = criterion_l1(fake_anime, anime_imgs) # Compare both original and newly created image
        loss_G = loss_G_GAN + 10.0 * loss_G_L1 # Total loss
        loss_G.backward() # Backpropogation
        optimizer_G.step()

        # ========================================
        # Logging
        # ========================================
        epoch_loss_G += loss_G.item()
        epoch_loss_D += loss_D.item()

        if (i + 1) % log_interval == 0:
            print(f"[Epoch {epoch}/{epochs}] [Batch {i+1}] "
                  f"Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

    # === Epoch Summary ===
    num_batches = len(anime_train_loader)
    print(f"→ Epoch {epoch} Done | Time: {time.time() - start_time:.1f}s | "
          f"Avg Loss_D: {epoch_loss_D / num_batches:.4f} | "
          f"Avg Loss_G: {epoch_loss_G / num_batches:.4f}")
    
    # === Save checkpoint every 5 epochs ===
    if epoch % 2 == 0:
        save_dir = f"checkpoints/epoch_{epoch}"
        os.makedirs(save_dir, exist_ok=True)
        torch.save(G.state_dict(), os.path.join(save_dir, "generator.pth"))
        torch.save(D.state_dict(), os.path.join(save_dir, "discriminator.pth"))


# from dataloader import get_dataloader
# anime_train_loader, real_train_loader = get_dataloader(anime_train, real_train)

# from torchvision.transforms.functional import to_pil_image
# batch = next(iter(anime_train_loader))
# image_tensor = batch[0]
# image_tensor = (image_tensor + 1) / 2
# pil_img = to_pil_image(image_tensor)
# pil_img.show()

