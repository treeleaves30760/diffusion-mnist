# Diffusion Model for MNIST

This repository contains a PyTorch implementation of a diffusion model applied to the MNIST dataset. The project includes:

- **Training** a diffusion model (UNet-like architecture) on MNIST
- **Generating** new handwritten digits based on the trained model
- **Evaluating** the model's performance using metrics like MSE and FID

## Table of Contents

- [Diffusion Model for MNIST](#diffusion-model-for-mnist)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Training](#training)
    - [Generating Samples](#generating-samples)
    - [Evaluation and Testing](#evaluation-and-testing)
  - [Results](#results)
  - [References](#references)
  - [License](#license)

---

## Overview

Diffusion models are a class of generative models that learn to reverse a forward noising process by progressively denoising samples. In this project, a simple UNet-like architecture is trained to predict the added noise at various time steps, allowing us to generate MNIST digits from random noise.

**Key points:**

- Uses the MNIST dataset (28×28 grayscale images).
- Implements a linear beta schedule for the diffusion process.
- Includes training (`train.py`), sampling (`generate.py`), and evaluation (`test.py`).

---

## Project Structure

```
.
├── checkpoints/               # Stores model checkpoints (saved every 10 epochs)
├── mnist/                     # Automatically downloaded MNIST data
├── datas/                     # Stores evaluation outputs (e.g., metrics plots)
├── src/
│   ├── train.py              # Script for training the diffusion model
│   ├── generate.py           # Script for generating images using a trained model
│   └── test.py               # Script for evaluating the model (MSE, FID)
└── README.md
```

- **`train.py`**  
  - Defines the UNet-like model (`SimpleUnet`).
  - Implements the training loop, forward diffusion, and noise prediction.
  - Saves checkpoints in `checkpoints/`.
  
- **`generate.py`**  
  - Loads a trained model checkpoint.
  - Performs the reverse diffusion process to generate new MNIST-like digits from random noise.
  - Saves generated samples as PNG images in `generated_samples/`.
  
- **`test.py`**  
  - Evaluates the trained model on the MNIST test set.
  - Computes Mean Squared Error (MSE) of the predicted noise versus actual noise.
  - Computes Fréchet Inception Distance (FID) (using features from the model’s latent representation).
  - Saves evaluation results and plots in `datas/`.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/treeleaves30760/diffusion-mnist.git
   cd diffusion-mnist
   ```

2. **Install required packages** (ensure you have Python 3.7+):
   ```bash
   pip install -r requirements.txt
   ```

   You also need to install [torch](https://pytorch.org/get-started/locally/) base on your environment.

---

## Usage

### Training

1. **Configure training parameters** in `train.py` or pass them as command-line arguments.  
   Common parameters include:
   - `--n_epochs`: Number of training epochs (default `1000`)
   - `--batch_size`: Mini-batch size (default `256`)
   - `--image_size`: Dimension of the MNIST images (default `28`)
   - `--channels`: Number of image channels (default `1` for MNIST)
   - `--time_steps`: Number of steps in the diffusion process (default `1000`)
   - `--device`: Device to use for training (`cuda` if available, otherwise `cpu`)

2. **Run the training script**:
   ```bash
   python src/train.py --n_epochs 1000 --batch_size 256 --time_steps 1000 --device cuda
   ```
   The script will:
   - Download the MNIST dataset (if not already downloaded) into `./mnist/`.
   - Train the UNet-like model for the specified number of epochs.
   - Save checkpoints every 10 epochs in the `checkpoints/` directory.

### Generating Samples

Once you have the trained model checkpoints (e.g., `diffusion_epoch_10.pth`, `diffusion_epoch_20.pth`, ...):

1. **Open `generate.py`** and ensure the parameters match your training configuration, or pass them as arguments if desired.
2. **Run the generation script**:
   ```bash
   python src/generate.py
   ```
   This will:
   - Load each checkpoint (by default, it tries loading `diffusion_epoch_{epoch}.pth` for `epoch` in steps of 10 until a max epoch).
   - Generate and save MNIST-like samples in the `generated_samples/` directory.
   - By default, it creates a file named `generated_samples_{epoch}.png` for each checkpoint.

*Note:* You may also specify a single checkpoint path by calling `generate_samples(model, checkpoint_path="path/to/checkpoint")` in the script or adjusting the code/arguments accordingly.

### Evaluation and Testing

To evaluate the model (MSE, FID) using the test set:

1. **Open `test.py`** to review or modify parameters.
2. **Run the evaluation**:
   ```bash
   python src/test.py
   ```
   The script will:
   - Load each checkpoint in increments of 10 epochs (`diffusion_epoch_10.pth`, `diffusion_epoch_20.pth`, etc.).
   - Calculate the MSE of predicted noise vs. actual noise on the MNIST test set.
   - Calculate the FID score based on learned features (using a simplified approach in the script).
   - Output plots (MSE vs. epoch, FID vs. epoch) and save them as `metrics.png` in `datas/`.

---

## Results

- **Training Loss**: Visual inspection of training loss provides an idea of how the model is learning to predict noise.  
- **Generated Samples**: You can find generated samples in `generated_samples/`.  
- **Evaluation Metrics**:  
  - **MSE**: Lower MSE indicates the model is better at predicting the noise.  
  - **FID**: Lower FID suggests higher quality and more realistic generated images (though for MNIST, FID might not be as standard as for more complex datasets).  

Example generated digits (after enough training epochs) might look like typical handwritten digits from the MNIST dataset.

---

## References

- **Diffusion Models**:  
  - [Denoising Diffusion Probabilistic Models (Ho et al., 2020)](https://arxiv.org/abs/2006.11239)  
- **UNet Architecture**:  
  - [U-Net: Convolutional Networks for Biomedical Image Segmentation (Ronneberger et al., 2015)](https://arxiv.org/abs/1505.04597)

- **Fréchet Inception Distance (FID)**:  
  - [Heusel et al., 2017](https://arxiv.org/abs/1706.08500)

---

## License

This project is open-sourced under the [MIT License](LICENSE). You are free to use, distribute, and modify this project as permitted under the terms of the license. If you use this project for academic research, please cite relevant sources.