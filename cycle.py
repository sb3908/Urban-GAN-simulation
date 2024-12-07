import argparse
import matplotlib.pyplot as plt
import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torch.amp import GradScaler
from torch.cuda.amp import autocast
from PIL import Image

# Directories for saving outputs
images_dir = 'C:/urban_simulation/Urban_simulation/generated_images'
log_dir = 'C:/urban_simulation/Urban_simulation/logs'
ckpt_dir = 'C:/urban_simulation/Urban_simulation/checkpoints'
os.makedirs(images_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(ckpt_dir, exist_ok=True)

# Force use of the first GPU
torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if not torch.cuda.is_available():
    raise EnvironmentError("CUDA is not available. Ensure GPU drivers and CUDA are properly installed.")
torch.cuda.set_device(device)
# Log CUDA device details
"""print(f"Using device: {device}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"Current CUDA device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(device)}")"""

# Define CycleGAN components
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(features),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=6):
        super(ResNetGenerator, self).__init__()
        self.num_residual_block = num_residual_blocks
        model = [
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        ]
        in_features = 64
        for _ in range(2):
            out_features = in_features * 2
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        for _ in range(num_residual_blocks):
            model += [ResidualBlock(in_features)]
        for _ in range(2):
            out_features = in_features // 2
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
        model += [nn.Conv2d(64, out_channels, kernel_size=7, stride=1, padding=3), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels):
        super(PatchGANDiscriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.Conv2d(512, 1, kernel_size=4, padding=1),
        )

    def forward(self, img):
        return self.model(img)


class UnpairedDataset(Dataset):
    def __init__(self, rootA, rootB, transform=None):
        self.files_A = sorted(glob.glob(os.path.join(rootA, "*.jpg")) + glob.glob(os.path.join(rootA, "*.png")))
        self.files_B = sorted(glob.glob(os.path.join(rootB, "*.jpg")) + glob.glob(os.path.join(rootB, "*.png")))
        self.transform = transform

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))

    def __getitem__(self, index):
        img_A = Image.open(self.files_A[index % len(self.files_A)]).convert("RGB")
        img_B = Image.open(self.files_B[index % len(self.files_B)]).convert("RGB")

        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}


# Training function
def train_cycle_gan(opt, dataloader, generator_A2B, generator_B2A, discriminator_A, discriminator_B, criterion_GAN, criterion_cycle, criterion_identity, optimizer_G, optimizer_D, scheduler_G, scheduler_D):
    scaler = GradScaler()
    real_label = 0.8  # Label smoothing
    fake_label = 0.2
    
    G_losses = []
    D_losses = []


    for epoch in range(opt.start_epoch, opt.n_epochs + 1):  # Start from the provided start_epoch
        G_loss_epoch = 0.0
        D_loss_epoch = 0.0
        for i, batch in enumerate(dataloader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # Train Generators
            optimizer_G.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                fake_B = generator_A2B(real_A)
                fake_A = generator_B2A(real_B)

                loss_GAN_A2B = criterion_GAN(discriminator_B(fake_B), torch.full_like(discriminator_B(fake_B), real_label, device=device))
                loss_GAN_B2A = criterion_GAN(discriminator_A(fake_A), torch.full_like(discriminator_A(fake_A), real_label, device=device))

                recov_A = generator_B2A(fake_B)
                recov_B = generator_A2B(fake_A)

                loss_cycle_A = criterion_cycle(recov_A, real_A) * 3.0
                loss_cycle_B = criterion_cycle(recov_B, real_B) * 3.0

                loss_identity_A = criterion_identity(fake_A, real_A) * 2.0
                loss_identity_B = criterion_identity(fake_B, real_B) * 2.0

                loss_G = loss_GAN_A2B + loss_GAN_B2A + loss_cycle_A + loss_cycle_B + loss_identity_A + loss_identity_B

            scaler.scale(loss_G).backward()
            torch.nn.utils.clip_grad_norm_(generator_A2B.parameters(), max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(generator_B2A.parameters(), max_norm=1.0)
            scaler.step(optimizer_G)
            scaler.update()

            # Train Discriminators less frequently
            if i % 3 == 0:
                optimizer_D.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss_real_A = criterion_GAN(discriminator_A(real_A), torch.full_like(discriminator_A(real_A), real_label, device=device))
                    loss_fake_A = criterion_GAN(discriminator_A(fake_A.detach()), torch.full_like(discriminator_A(fake_A.detach()), fake_label, device=device))
                    loss_D_A = (loss_real_A + loss_fake_A) * 3

                    loss_real_B = criterion_GAN(discriminator_B(real_B), torch.full_like(discriminator_B(real_B), real_label, device=device))
                    loss_fake_B = criterion_GAN(discriminator_B(fake_B.detach()), torch.full_like(discriminator_B(fake_B.detach()), fake_label, device=device))
                    loss_D_B = (loss_real_B + loss_fake_B) * 3

                scaler.scale(loss_D_A).backward()
                scaler.scale(loss_D_B).backward()

                torch.nn.utils.clip_grad_norm_(discriminator_A.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(discriminator_B.parameters(), max_norm=1.0)

                scaler.step(optimizer_D)
                scaler.update()

            if i % 100 == 0:
                print(f"[Epoch {epoch}/{opt.n_epochs}] [Batch {i}/{len(dataloader)}] [D loss: {(loss_D_A + loss_D_B).item():.4f}] [G loss: {loss_G.item():.4f}]")
            G_loss_epoch += loss_G.item()
            D_loss_epoch += (loss_D_A + loss_D_B).item()

        G_losses.append(G_loss_epoch / len(dataloader))  # Average G loss per epoch
        D_losses.append(D_loss_epoch / len(dataloader))  # Average D loss per epoch

        scheduler_G.step()
        scheduler_D.step()

        # Save models
        torch.save(generator_A2B.state_dict(), os.path.join(ckpt_dir, f"generator_A2B_epoch_{epoch}.pth"))
        torch.save(generator_B2A.state_dict(), os.path.join(ckpt_dir, f"generator_B2A_epoch_{epoch}.pth"))
        torch.save(discriminator_A.state_dict(), os.path.join(ckpt_dir, f"discriminator_A_epoch_{epoch}.pth"))
        torch.save(discriminator_B.state_dict(), os.path.join(ckpt_dir, f"discriminator_B_epoch_{epoch}.pth"))
        print(f"Models saved for epoch {epoch}")

    print("Training complete!")

    save_losses(G_losses, D_losses)

    return G_losses, D_losses
def save_losses(g_losses, d_losses, filename="losses.txt"):
    with open(filename, "w") as f:
        f.write("Generator Losses:\n")
        for loss in g_losses:
            f.write(f"{loss}\n")
        f.write("\nDiscriminator Losses:\n")
        for loss in d_losses:
            f.write(f"{loss}\n")

# Main execution
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=50, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
   # parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--decay_epoch", type=int, default=45, help="epoch to start linearly decaying the learning rate")
    parser.add_argument("--n_cpu", type=int, default=16, help="number of cpu threads to use during batch generation")
    parser.add_argument("--start_epoch", type=int, default=25, help="epoch to start training from")
    parser.add_argument("--b1", type=float, default=0.5, help="Beta1 for Adam optimizer")
    parser.add_argument("--b2", type=float, default=0.999, help="Beta2 for Adam optimizer")
    parser.add_argument("--lambda_identity", type=float, default=0.5, help="Identity loss weight")
    parser.add_argument("--lambda_cycle", type=float, default=10.0, help="Cycle consistency loss weight")
    parser.add_argument("--input_shape", type=tuple, default=(64, 64), help="input image dimensions")
    opt = parser.parse_args()

    # Initialize models
    generator_A2B = ResNetGenerator(3, 3,num_residual_blocks=6).to("cuda")
    generator_B2A = ResNetGenerator(3, 3,num_residual_blocks=6).to("cuda")
    discriminator_A = PatchGANDiscriminator(3).to("cuda")
    discriminator_B = PatchGANDiscriminator(3).to("cuda")

    checkpoint_path_A2B = "C:/urban_simulation/Urban_simulation/checkpoints/generator_A2B_epoch_25.pth"
    checkpoint_path_B2A = "C:/urban_simulation/Urban_simulation/checkpoints/generator_B2A_epoch_25.pth"
    checkpoint_path_dis_A = "C:/urban_simulation/Urban_simulation/checkpoints/discriminator_A_epoch_25.pth"
    checkpoint_path_dis_B = "C:/urban_simulation/Urban_simulation/checkpoints/discriminator_B_epoch_25.pth"

    generator_A2B.load_state_dict(torch.load(checkpoint_path_A2B, map_location=device))
    generator_B2A.load_state_dict(torch.load(checkpoint_path_B2A, map_location=device))
    discriminator_A.load_state_dict(torch.load(checkpoint_path_dis_A, map_location=device))
    discriminator_B.load_state_dict(torch.load(checkpoint_path_dis_B, map_location=device))

    # Load saved weights if resuming
    if opt.start_epoch > 1:
        generator_A2B.load_state_dict(torch.load(os.path.join(ckpt_dir, f"generator_A2B_epoch_{opt.start_epoch - 1}.pth")))
        generator_B2A.load_state_dict(torch.load(os.path.join(ckpt_dir, f"generator_B2A_epoch_{opt.start_epoch - 1}.pth")))
        discriminator_A.load_state_dict(torch.load(os.path.join(ckpt_dir, f"discriminator_A_epoch_{opt.start_epoch - 1}.pth")))
        discriminator_B.load_state_dict(torch.load(os.path.join(ckpt_dir, f"discriminator_B_epoch_{opt.start_epoch - 1}.pth")))
        print(f"Resumed training from epoch {opt.start_epoch}")

    # Loss functions
    criterion_GAN = nn.MSELoss()
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # Optimizers and schedulers
    optimizer_G = optim.Adam(
        itertools.chain(generator_A2B.parameters(), generator_B2A.parameters()), lr=0.0002, betas=(opt.b1, opt.b2)
    )
    optimizer_D = optim.Adam(
        itertools.chain(discriminator_A.parameters(), discriminator_B.parameters()), lr=0.0003, betas=(opt.b1, opt.b2)
    )
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda epoch: 1.0 - max(0, epoch - opt.decay_epoch) / float(opt.n_epochs - opt.decay_epoch))
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda epoch: 1.0 - max(0, epoch - opt.decay_epoch) / float(opt.n_epochs - opt.decay_epoch))


    # Load dataset
    transform = transforms.Compose([
        transforms.Resize(opt.input_shape),
         transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    dataset = UnpairedDataset(
        rootA='C:/urban_simulation/Urban_simulation/image_data',
        rootB='C:/urban_simulation/Urban_simulation/image2',
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu)

    # Train CycleGAN
    train_cycle_gan(opt, dataloader, generator_A2B, generator_B2A, discriminator_A, discriminator_B, criterion_GAN, criterion_cycle, criterion_identity, optimizer_G, optimizer_D, scheduler_G, scheduler_D)
