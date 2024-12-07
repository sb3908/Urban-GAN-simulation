import matplotlib.pyplot as plt

# Function to load losses from a file
def load_losses(filename="losses.txt"):
    g_losses = []
    d_losses = []
    
    with open(filename, "r") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]  # Remove blank lines
        
        if "Generator Losses:" not in lines or "Discriminator Losses:" not in lines:
            raise ValueError("The losses.txt file is missing required headers (Generator Losses or Discriminator Losses).")
        
        g_losses_start = lines.index("Generator Losses:") + 1
        d_losses_start = lines.index("Discriminator Losses:") + 1
        
        try:
            g_losses = [float(line) for line in lines[g_losses_start:d_losses_start-1]]
            d_losses = [float(line) for line in lines[d_losses_start:]]
        except ValueError as e:
            raise ValueError(f"Invalid loss value found in {filename}: {e}")
    
    return g_losses, d_losses


# Function to plot the losses
def plot_losses(g_losses, d_losses):
    plt.figure(figsize=(10, 5))

    # Plot Generator Loss
    plt.subplot(1, 2, 1)
    plt.plot(g_losses, label="Generator Loss", color="blue")
    plt.title("Generator Loss During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    # Plot Discriminator Loss
    plt.subplot(1, 2, 2)
    plt.plot(d_losses, label="Discriminator Loss", color="red")
    plt.title("Discriminator Loss During Training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    g_losses, d_losses = load_losses("losses.txt")  # Load the losses from the file
    plot_losses(g_losses, d_losses)  # Plot the losses
