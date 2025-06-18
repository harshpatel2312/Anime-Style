import torch
import matplotlib.pyplot as plt
from generator import Generator  # adjust the path if needed
from torchvision.transforms.functional import to_pil_image

# 1. Create a dummy input: batch size 1, 3 channels, 512x512
dummy_input = torch.randn(1, 3, 512, 512)

# 2. Initialize the generator
gen = Generator()

# 3. Set to eval mode (optional, disables dropout/batchnorm randomness)
gen.eval()

# 4. Run the dummy image through the generator
with torch.no_grad():
    output = gen(dummy_input)

print("Output shape:", output.shape)
print("Output range:", output.min().item(), "to", output.max().item())

# 5. Convert output to image (denormalize from [-1, 1] to [0, 1])
img_tensor = (output[0] + 1) / 2  # remove batch, rescale

# 6. Convert to PIL and show
img_pil = to_pil_image(img_tensor.clamp(0, 1))  # ensure range is valid
img_pil.show()
