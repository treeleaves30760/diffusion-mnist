import torch
import matplotlib.pyplot as plt
from train import SimpleUnet, device, channels, image_size, time_steps
import os
import argparse
from tqdm import tqdm


def get_index_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def linear_beta_schedule(timesteps):
    beta_start = 0.0001
    beta_end = 0.02
    return torch.linspace(beta_start, beta_end, timesteps)

@torch.no_grad()
def sample_timestep(x, t, model, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas):
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    
    if t[0] == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(betas_t) * noise

def generate_samples(model, epoch=None, n_samples=16, checkpoint_path=None, save_dir="generated_samples"):
    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.eval()
    
    # Calculate diffusion parameters
    betas = linear_beta_schedule(time_steps)
    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, 0)
    sqrt_recip_alphas = torch.sqrt(1 / alphas)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
    
    # Create random noise
    x = torch.randn(n_samples, channels, image_size, image_size).to(device)
    
    # Generate images
    print("Generating images...")
    for t in tqdm(reversed(range(time_steps))):
        t_batch = torch.tensor([t] * n_samples, device=device)
        x = sample_timestep(x, t_batch, model, betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas)
    
    # Create directory for saving generated images
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot and save generated images
    plt.figure(figsize=(10, 10))
    for i in range(n_samples):
        plt.subplot(int(n_samples**0.5), int(n_samples**0.5), i + 1)
        plt.imshow(x[i].reshape(image_size, image_size).cpu().numpy(), cmap='gray')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'generated_samples_{epoch}.png'), pad_inches=0.1)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--n_samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1000)
    args = parser.parse_args()
    
    model = SimpleUnet(channels=args.channels).to(device)
    
    epochs = args.epochs
    for epoch in range(10, epochs + 1, 10):
        checkpoint_path = f"checkpoints/diffusion_epoch_{epoch}.pth"
        generate_samples(model, epoch=epoch, checkpoint_path=checkpoint_path)