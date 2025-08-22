# Anime-Style GAN 🎨

This project explores **image-to-image translation** for converting real-world photos into anime-style images using a **Generative Adversarial Network (GAN)**.  
It is inspired by architectures like **CycleGAN**, using a ResNet-based Generator and PatchGAN Discriminator.

> ⚠️ **Work in Progress**: The project is under active development. Training and loss functions are still being refined.

---

## ✨ Features
- **Custom Dataset Loader** for anime and real images (supports `.png`, `.jpg`, `.jpeg`).
- **Generator**: ResNet-based with downsampling, residual blocks, and upsampling.
- **Discriminator**: PatchGAN classifier for local realism.
- **Losses**: GAN loss (LSGAN) + L1 reconstruction (placeholder, will refine for unpaired setting).
- **Training Pipeline** with checkpointing and resume support.
- **Testing Script** to load a trained generator and stylize new images.

---

## 📂 Project Structure
```
Anime-Style/
│── model/
    │── config.py # Dataset definition & transforms
    │── dataloader.py # DataLoader setup for anime & real images
    │── discriminator.py # PatchGAN discriminator
    │── generator.py # ResNet-based generator
    │── generator-debugging.py # Debugging script with dummy input
    │── resnet_block.py # Residual block implementation
    │── losses.py # GAN loss (MSE/BCE)
    │── train.py
    │── test.py
    │── checkpoints/ # Saved model checkpoints (auto-created)

│── data/
    ├── anime/train/ # Anime images
    │── real/train/ # Real images
```

---

## ⚙️ Environment Setup and Dependency Requirements
### Virtual Enviornment:
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### Dependencies:
- Python 3.12
- torch (PyTorch)
- torchvision
- opencv-python
- tqdm
- matplotlib (for debugging)

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 🚀 Training
> ⚠️ Dataset isn't available at the moment. It will be uploaded shortly...
1. Prepare datasets:
  ```bash
  data/
  ├── anime/train/
  └── real/train/
  ```

2. Run training:
  ```bash
  python train.py
  ```
  - Checkpoints will be saved every 100 batches under:
    ```bash
    checkpoints/epoch_X_batch_Y/checkpoint.pth
    ```

3. Resume training is automatic if checkpoints exist.

---

## 🎨 Testing
Use the provided script to stylize an image with a trained generator:
```bash
python test.py
```

This will:
- Load a generator from the latest checkpoint.
- Process a test image (Test-Pics/...).
- Display the stylized anime-style result.

---

## 🔧 Notes
- Current training uses L1 loss against random anime samples. This works for debugging but is not ideal for unpaired datasets — expect blurry outputs.
- Next steps:
  - Introduce cycle-consistency loss (CycleGAN).
  - Add identity/perceptual losses for better content preservation.
  - Fine-tune learning rates and checkpoint strategy.

---

## 📃 License
MIT License © Harsh Patel
