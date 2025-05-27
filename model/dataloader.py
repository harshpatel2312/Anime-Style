from dataset import AnimeStyleDataset
from torch.utils.data import DataLoader

def get_dataloader(anime_train, real_train):
    # Create on-the-fly datasets with preprocessed images
    anime_train_dataset = AnimeStyleDataset(anime_train)
    real_train_dataset = AnimeStyleDataset(real_train)

    # Create DataLoaders
    anime_train_loader = DataLoader(anime_train_dataset, batch_size = 8, shuffle = True, num_workers = 2)
    real_train_loader = DataLoader(real_train_dataset, batch_size = 8, shuffle = True, num_workers = 2)

    return anime_train_loader, real_train_loader
