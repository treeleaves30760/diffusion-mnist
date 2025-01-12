import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import argparse

class SimpleUnet(nn.Module):
    def __init__(self, channels=1):
        super().__init__()
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU()
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 + 256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.SiLU()
        )
        
        # Decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(256 + 128, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU()
        )
        self.final = nn.Conv2d(64 + 64, channels, 3, padding=1)
        
    def forward(self, x, t):
        # Time embedding
        t = t.unsqueeze(-1).float()
        t = self.time_mlp(t)
        
        # Encoder
        x1 = self.enc1(x)           # [B, 64, 28, 28]
        x2 = self.enc2(x1)          # [B, 128, 14, 14]
        x3 = self.enc3(x2)          # [B, 256, 7, 7]
        
        # Bottleneck
        # Expand time embedding and concatenate
        t = t.view(-1, 256, 1, 1).expand(-1, -1, x3.shape[2], x3.shape[3])
        xb = self.bottleneck(torch.cat([x3, t], 1))  # [B, 512, 7, 7]
        
        # Decoder with skip connections
        x = self.dec3(xb)                        # [B, 256, 14, 14]
        x = self.dec2(torch.cat([x, x2], 1))     # [B, 64, 28, 28]
        x = self.final(torch.cat([x, x1], 1))    # [B, 1, 28, 28]
        
        return x

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, betas, alphas_cumprod):
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    x_noisy = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape) * x_0 + \
              get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
    return x_noisy, noise

def get_data(batch_size=256):
    transforms_train = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root="./mnist/", train=True, download=True, transform=transforms_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    return dataloader

def train_one_epoch(model, dataloader, optimizer, epoch, betas, alphas_cumprod, device="cuda", time_steps=1000):
    model.train()
    total_loss = 0
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch}")
    
    for step, (images, _) in progress_bar:
        optimizer.zero_grad()
        
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, time_steps, (batch_size,), device=device).long()
        
        # Get noisy image and noise
        x_noisy, noise = forward_diffusion_sample(images, t, betas, alphas_cumprod)
        
        # Get noise prediction
        noise_pred = model(x_noisy, t)
        
        # Calculate loss
        loss = F.mse_loss(noise_pred, noise)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})
    
    return total_loss / len(dataloader)

def main(args: argparse.Namespace):
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    device = args.device
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    image_size = args.image_size
    channels = args.channels
    time_steps = args.time_steps
    
    # Initialize model and optimizer
    model = SimpleUnet(channels=channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Get data
    dataloader = get_data(batch_size=batch_size)
    
    # Calculate necessary values for diffusion process
    betas = linear_beta_schedule(time_steps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    
    # Training loop
    for epoch in range(n_epochs):
        loss = train_one_epoch(model, dataloader, optimizer, epoch, betas, alphas_cumprod, device=device, time_steps=time_steps)
        print(f"Epoch {epoch} average loss: {loss:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }
            torch.save(checkpoint, f"checkpoints/diffusion_epoch_{epoch+1}.pth")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--time_steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)