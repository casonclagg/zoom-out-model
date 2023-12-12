import torch
from torchvision import transforms
from PIL import Image
import model  # import the UNet model from model.py
import argparse
import os
import cv2

import shutil

# Function to clear contents of the results directory
def clear_results_directory(directory):
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

clear_results_directory('results')

# Example Usage:
# python infer.py models/v2_epoch_5.pth input.png epoch_5 --iterations 120


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = tensor * std + mean  # Denormalize
    tensor = torch.clamp(tensor, 0, 1)  # Clamp values to the range [0, 1]
    return tensor


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Run inference with a UNet model checkpoint.')
parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint file.')
parser.add_argument('input', type=str, help='input image')
parser.add_argument('output', type=str, help='base name for output video')
parser.add_argument('--iterations', type=int, default=1, help='Number of iterations to feed output back as input')
args = parser.parse_args()

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the trained model
net = model.UNet()
checkpoint = torch.load(args.checkpoint, map_location=device)
net.load_state_dict(checkpoint['model_state_dict'])
net.to(device)s
net.eval()

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Function to save image
def save_image(tensor, filename):
    output_image = transforms.ToPILImage()(tensor)
    output_image.save(f'results/{filename}')

# Load initial image
image = Image.open(args.input).convert('RGB')
# Resize and crop to 256x256
image = transforms.CenterCrop(256)(image)
image.save('input_image.png')
input_tensor = transform(image).unsqueeze(0).to(device)

# Perform iterations and save images
for i in range(args.iterations):
    with torch.no_grad():
        output = net(input_tensor)
        output = output.squeeze(0).cpu()

    # Save output image
    filename = f'{args.output}_{i:05}.png'
    output = denormalize(output)
    save_image(output, filename)

    # Use output as next input
    input_tensor = output.unsqueeze(0).to(device)

# Function to create a video from images
def create_video(image_folder, video_name, fps=10):
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    images.sort()  # Ensure the images are in order

    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()

# Create video from saved images
create_video('results', f'{args.output}.mp4')
