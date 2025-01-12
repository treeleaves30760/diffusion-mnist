import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from train import SimpleUnet
import numpy as np
from tqdm import tqdm
import argparse

def get_test_data(batch_size=256):
    transforms_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.MNIST(root="./mnist/", train=False, download=True, transform=transforms_test)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def calculate_fid(real_features, generated_features, device="cuda"):
    # Convert numpy arrays to PyTorch tensors if they aren't already
    if isinstance(real_features, np.ndarray):
        real_features = torch.from_numpy(real_features)
    if isinstance(generated_features, np.ndarray):
        generated_features = torch.from_numpy(generated_features)
    
    # Move to device if not already there
    real_features = real_features.to(device)
    generated_features = generated_features.to(device)
    
    # Calculate mean and covariance for real and generated features
    mu_real = torch.mean(real_features, dim=0)
    mu_gen = torch.mean(generated_features, dim=0)
    
    sigma_real = torch.cov(real_features.T)
    sigma_gen = torch.cov(generated_features.T)
    
    # Calculate squared difference between means
    diff = mu_real - mu_gen
    diff_squared = torch.matmul(diff, diff)
    
    # Calculate matrix trace using matrix square root
    # Using torch.linalg.eigvalsh for numerical stability
    product = torch.matmul(sigma_real, sigma_gen)
    eigvals = torch.linalg.eigvalsh(product)
    # Handle negative eigenvalues that might occur due to numerical instability
    eigvals = torch.where(eigvals < 0, torch.zeros_like(eigvals), eigvals)
    covmean = torch.sqrt(eigvals).sum()
    
    # Calculate trace terms
    trace_real = torch.trace(sigma_real)
    trace_gen = torch.trace(sigma_gen)
    
    # Calculate FID
    fid = diff_squared + trace_real + trace_gen - 2 * covmean
    
    return fid.item()

@torch.no_grad()
def evaluate_model(model, test_dataloader, checkpoint_path, device="cuda", time_steps=1000):
    print(f"Loading checkpoint from {checkpoint_path}")
    # Load model checkpoint
    checkpoint = torch.load(checkpoint_path, weights_only=True)  # Set weights_only=True for security
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    total_mse = 0
    real_features = []
    generated_features = []
    
    print("Evaluating model...")
    for images, _ in tqdm(test_dataloader):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, time_steps, (batch_size,), device=device).long()
        
        # Add noise
        noise = torch.randn_like(images)
        noisy_images = images + noise * (t.view(-1, 1, 1, 1) / time_steps).sqrt()
        
        # Predict noise
        noise_pred = model(noisy_images, t)
        
        # Calculate MSE
        mse = F.mse_loss(noise_pred, noise)
        total_mse += mse.item()
        
        # Extract features for FID calculation
        real_features.append(images.view(batch_size, -1))
        generated_features.append(noisy_images.view(batch_size, -1))
    
    # Calculate metrics
    avg_mse = total_mse / len(test_dataloader)
    
    # Concatenate all features
    real_features = torch.cat(real_features, dim=0)
    generated_features = torch.cat(generated_features, dim=0)
    
    print("Calculating FID score...")
    fid_score = calculate_fid(real_features, generated_features, device=device)
    
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"FID Score: {fid_score:.4f}")
    
    return avg_mse, fid_score

def main(args: argparse.Namespace):
    
    channels = args.channels
    device = args.device
    time_steps = args.time_steps
    batch_size = args.batch_size
    epochs = args.epochs
    
    # Initialize model
    model = SimpleUnet(channels=channels).to(device)
    test_dataloader = get_test_data(batch_size=batch_size)
    
    # Evaluate checkpoints
    avg_mes_list = []
    fid_scores_list = []
    epochs_list = []
    epochs = 200
    for i in range(10, epochs + 1, 10):
        checkpoint_path = f"checkpoints/diffusion_epoch_{i}.pth"
        avg_mse, fid_score = evaluate_model(model, test_dataloader, checkpoint_path, device=device, time_steps=time_steps)
        epochs_list.append(i)
        avg_mes_list.append(avg_mse)
        fid_scores_list.append(fid_score)
        
    # Plot metrics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_list, avg_mes_list, marker='o')
    plt.title("Average MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_list, fid_scores_list, marker='o')
    plt.title("FID Score")
    plt.xlabel("Epoch")
    plt.ylabel("FID")
    
    plt.tight_layout()
    plt.savefig(os.path.join("datas", "metrics.png"))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--time_steps", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    main(args)