# models/training_scripts/train_makeup_model.py

import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# Your custom model class (we'll define it in model_utils.py)
from models.model_utils import MakeupApplicationModel

# Paths
MAKEUP_DIR = "data/makeup_ref"
NOMAKEUP_DIR = "data/no_makeup"
SAVE_MODEL_PATH = "models/trained_weights/makeup_gan.pth"
OUTPUT_SAMPLE_PATH = "outputs/"

# Basic transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Dataset setup (paired images)
class MakeupDataset(torch.utils.data.Dataset):
    def __init__(self, nomakeup_dir, makeup_dir, transform=None):
        self.nomakeup_imgs = sorted(os.listdir(nomakeup_dir))
        self.makeup_imgs = sorted(os.listdir(makeup_dir))
        self.transform = transform
        self.nomakeup_dir = nomakeup_dir
        self.makeup_dir = makeup_dir

    def __len__(self):
        return min(len(self.nomakeup_imgs), len(self.makeup_imgs))

    def __getitem__(self, idx):
        nomakeup_path = os.path.join(self.nomakeup_dir, self.nomakeup_imgs[idx])
        makeup_path = os.path.join(self.makeup_dir, self.makeup_imgs[idx])

        nomakeup = Image.open(nomakeup_path).convert("RGB")
        makeup = Image.open(makeup_path).convert("RGB")

        if self.transform:
            nomakeup = self.transform(nomakeup)
            makeup = self.transform(makeup)

        return nomakeup, makeup

# Train function
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = MakeupDataset(NOMAKEUP_DIR, MAKEUP_DIR, transform)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = MakeupApplicationModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)

    epochs = 10

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for i, (no_makeup, makeup) in enumerate(loop):
            no_makeup = no_makeup.to(device)
            makeup = makeup.to(device)

            loss = model.train_step(no_makeup, makeup, optimizer)

            loop.set_postfix(loss=loss)

        # Save one sample output
        sample_output = model(no_makeup)
        save_image(sample_output, os.path.join(OUTPUT_SAMPLE_PATH, f"sample_epoch_{epoch+1}.png"))

    torch.save(model.state_dict(), SAVE_MODEL_PATH)

if __name__ == "__main__":
    train()
