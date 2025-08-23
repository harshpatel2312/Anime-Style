# test.py
from generator import Generator
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- paths ---
checkpoint_path = "/content/drive/MyDrive/Colab Notebooks/Anime-Style/checkpoints/epoch_5_batch_500/checkpoint.pth"
test_image_path = "/content/drive/MyDrive/Colab Notebooks/Anime-Style/Test-Pics/pexels-olly-733872.jpg"
out_dir = "/content/drive/MyDrive/Colab Notebooks/Anime-Style/Outputs"
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "stylized.png")

# --- model ---
G = Generator(input_nc=3, output_nc=3, ngf=64, n_blocks=6).to(device)
ckpt = torch.load(checkpoint_path, map_location=device)
G.load_state_dict(ckpt["G_state_dict"])
G.eval()

# --- preprocess ---
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # [-1,1]
])

img = Image.open(test_image_path).convert("RGB")
inp = transform(img).unsqueeze(0).to(device)

# --- inference ---
with torch.no_grad():
    out = G(inp)  # [-1,1]

print("Output stats (pre-denorm):",
      f"min={out.min().item():.4f}, max={out.max().item():.4f},",
      f"mean={out.mean().item():.4f}, std={out.std().item():.4f}")

# --- save ---
out_vis = (out.squeeze(0).cpu() + 1) / 2.0
out_vis = out_vis.clamp(0, 1)
to_pil_image(out_vis).save(out_path)
print(f"Saved stylized image to: {out_path}")
