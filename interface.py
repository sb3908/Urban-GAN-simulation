import streamlit as st
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Normalize, Resize, ToPILImage
from cycle import ResNetGenerator

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# Function to load the model
@st.cache_resource
def load_model(checkpoint_path):
    model = ResNetGenerator(3, 3).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model

# Transformations for input and output images
transform_input = Compose([
    Resize((128, 128)),
    ToTensor(),
    Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
])

transform_output = Compose([
    ToPILImage(),
    Resize((128, 128))  # Resize for display purposes
])

# Main app logic
def main():
    st.title("CycleGAN Image Transformation")
    st.write("Upload an image to apply CycleGAN transformations.")

    # Sidebar options
    st.sidebar.title("Settings")
    model_choice = st.sidebar.radio(
        "Select Transformation Direction:", 
        ["A to B", "B to A"]
    )
    
    # Path to model checkpoints based on user choice
    checkpoint_path = (
        "C:/urban_simulation/Urban_simulation/checkpoints/generator_A2B_epoch_100.pth" 
        if model_choice == "A to B" 
        else "C:/urban_simulation/Urban_simulation/checkpoints/generator_B2A_epoch_100.pth"
    )

    # Load the chosen model
    generator = load_model(checkpoint_path)

    # File uploader for input image
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png"])
    if uploaded_file is not None:
        # Display the uploaded image
        st.subheader("Uploaded Image")
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Original Image", use_column_width=True)

        # Preprocess the image and run inference
        st.subheader("Generated Image")
        with st.spinner("Processing..."):
            input_tensor = transform_input(input_image).unsqueeze(0).to(device)  # Add batch dimension and move to device
            with torch.no_grad():
                output_tensor = generator(input_tensor).squeeze(0).cpu()  # Move back to CPU for processing
                output_image = transform_output(output_tensor)

        # Display the transformed image
        st.image(output_image, caption="Transformed Image", use_column_width=True)

        # Option to download the generated image
        save_option = st.button("Download Generated Image")
        if save_option:
            output_path = "transformed_image.png"
            output_image.save(output_path)
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Transformed Image",
                    data=file,
                    file_name="transformed_image.png",
                    mime="image/png"
                )

if __name__ == "__main__":
    main()
