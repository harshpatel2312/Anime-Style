from generator import Generator
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image

# Load the Generator model with same saved weights
G = Generator(input_nc = 3, output_nc = 3, ngf = 64, n_blocks = 6)

# Load weights
checkpoint_path = "checkpoints/epoch_1_batch_400/generator.pth"
G.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
G.eval()
G.to("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess the test image
img = Image.open(r"C:\Users\harsh\Downloads\pexels-olly-733872.jpg")

# Preprocessing image
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Add batch dimension
input_tensor = transform(img).unsqueeze(0).to("cuda" if torch.cuda.is_available() else "cpu")

# Anime Generation
with torch.no_grad():
    output_tensor = G(input_tensor)

# Denormalize
output_tensor = (output_tensor.squeeze(0).cpu() + 1) / 2.0
print(output_tensor)
print(output_tensor.shape)

output_image = to_pil_image(output_tensor)
output_image.show()