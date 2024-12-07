# ssim_utils.py
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure

def initialize_ssim(device):
    return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

def compute_ssim(real_images, generated_images, ssim_metric):
    """
    Compute SSIM for a batch of real and generated images.
    :param real_images: Tensor of real images
    :param generated_images: Tensor of generated images
    :param ssim_metric: Initialized SSIM metric object
    :return: Mean SSIM score for the batch
    """
    ssim_scores = []
    for real, gen in zip(real_images, generated_images):
        real = real.unsqueeze(0)  # Add batch dimension
        gen = gen.unsqueeze(0)    # Add batch dimension
        ssim_scores.append(ssim_metric(real, gen).item())
    return np.mean(ssim_scores)

def visualize_ssim(ssim_scores, save_path="ssim_plot.png"):
    """
    Visualize SSIM scores over epochs.
    :param ssim_scores: List of SSIM scores for each epoch
    :param save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(ssim_scores, label="SSIM Score")
    plt.xlabel("Epochs")
    plt.ylabel("SSIM")
    plt.title("SSIM Score Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
