import torch
import torch.nn as nn

class GANLoss(nn.Module):
    def __init__(self, use_mse = True, target_real_label = 1.0, target_fake_label = 0.0):
        super(GANLoss, self).__init__()
        """
        Args:
            use_mse (bool): Use MSE loss instead of BCE.
            target_real_label (float): Label for real images.
            target_fake_label (float): Label for fake images.
        """
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.loss = nn.MSELoss() if use_mse else nn.BCEWithLogitsLoss()

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            return torch.full_like(prediction, self.real_label)
        else:
            return torch.full_like(prediction, self.fake_label)

    def forward(self, prediction, target_is_real):
        """
        Compute GAN loss: real vs fake.
        """
        target_tensor = self.get_target_tensor(prediction, target_is_real)
        return self.loss(prediction, target_tensor)