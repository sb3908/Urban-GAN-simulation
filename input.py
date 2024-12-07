import torch
import torchvision.transforms as transforms
from PIL import Image
import os
from cycle import ResNetGenerator  # Assuming you saved your model definitions here

# Paths
input_dir = 'C:/urban_simulation/Urban_simulation/image_data'  # Directory containing input images from Domain A
output_dir = 'C:/urban_simulation/Urban_simulation/generated_images'  # Directory to save output images in Domain B style
os.makedirs(output_dir, exist_ok=True)

# Load the trained model
device = "cuda" if torch.cuda.is_available() else "cpu"
generator_A2B = ResNetGenerator(3, 3).to(device)
generator_A2B.load_state_dict(torch.load("C:/urban_simulation/Urban_simulation/checkpoints/generator_A2B_epoch_100.pth", map_location=device))
generator_A2B.eval()  # Set the model to evaluation mode

# Define transformation for the input images
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Inference
def generate_images(input_dir, output_dir, generator, transform, device):
    input_images = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith((".jpg", ".png"))]

    for img_path in input_images:
        # Load and preprocess the image
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Generate the output image
        with torch.no_grad():
            fake_img_tensor = generator(img_tensor)
        
        # Post-process the output image (convert back to image format)
        fake_img = fake_img_tensor.squeeze(0).cpu()
        fake_img = (fake_img * 0.5 + 0.5).clamp(0, 1)  # Denormalize
        fake_img = transforms.ToPILImage()(fake_img)

        # Save the output image
        output_path = os.path.join(output_dir, os.path.basename(img_path))
        fake_img.save(output_path)
        print(f"Saved generated image to {output_path}")

# Run the inference
generate_images(input_dir, output_dir, generator_A2B, transform, device)
