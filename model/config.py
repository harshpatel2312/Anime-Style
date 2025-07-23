from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class AnimeStyleDataset(Dataset):
    def __init__(self, root_dir):
        """
        Args: 
            root_dir : Path to the image folder (e.g, anime/train, real/train)
        """
        self.root_dir = root_dir
        self.image_paths = [os.path.join(root_dir, file) 
                            for file in os.listdir(root_dir) if file.lower().endswith((".png", ".jpeg", "jpg"))]

        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            #transforms.Lambda(lambda img: transforms.functional.crop(img, top=0, left=0, height=256, width=256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image
    