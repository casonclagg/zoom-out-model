import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as vutils
from torchvision import transforms
from torch.utils.data import DataLoader
import model  # import the UNet model from model.py
from dataset import ImagePairDataset  # replace with your actual file name
from pytorch_msssim import SSIM  # Assuming you have installed a package like pytorch-msssim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from torch.utils.tensorboard import SummaryWriter

# Create a TensorBoard writer
MODEL_NAME="v3"
writer = SummaryWriter(f'runs/{MODEL_NAME}')




print(torch.__version__)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = ImagePairDataset(root_dir='dataset', transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Initialize the model
net = model.UNet().to(device)

# Loss and optimizer
mse_loss = nn.MSELoss()
ssim_loss = SSIM(data_range=255, size_average=True, channel=3)

def edge_loss(y_pred, y_true):
    # Define the edge filter
    edge_filter = torch.tensor([[-1, -1, -1],
                                [-1, 8, -1],
                                [-1, -1, -1]], dtype=y_pred.dtype).view(1, 1, 3, 3).to(y_pred.device)

    # Repeat the filter for each input channel
    edge_filter = edge_filter.repeat(y_pred.size(1), 1, 1, 1)

    # Apply group convolution to apply the filter to each channel independently
    y_pred_edges = F.conv2d(y_pred, edge_filter, groups=y_pred.size(1), padding=1)
    y_true_edges = F.conv2d(y_true, edge_filter, groups=y_true.size(1), padding=1)

    # Calculate the loss
    return mse_loss(y_pred_edges, y_true_edges)


# During training, compute combined loss
def combined_loss(y_pred, y_true, alpha=0.5, beta=0.5):
    mse = mse_loss(y_pred, y_true)
    ssim = 1 - ssim_loss(y_pred, y_true)  # SSIM loss is often defined as (1 - SSIM)
    edge = edge_loss(y_pred, y_true)

    # Combine the losses
    combined = mse + alpha * ssim + beta * edge
    return combined

optimizer = optim.Adam(net.parameters(), lr=0.01)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

# Load checkpoint if exists
checkpoint_path = f'models/{MODEL_NAME}_epoch_65.pth'
start_epoch = 0
if os.path.isfile(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f'Loaded checkpoint from epoch {start_epoch}')

def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Unnormalize a tensor image."""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # Unnormalize
    return tensor

# Train the model
num_epochs = 501
save_checkpoint_every = 5
log_image_interval = 2
n_images = 1

# checkpoint_filename = f'models/{MODEL_NAME}_epoch_0.pth'
# torch.save({
#     'epoch': 0,
#     'model_state_dict': net.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'scheduler_state_dict': scheduler.state_dict(),
# }, checkpoint_filename)

for epoch in range(start_epoch, num_epochs):
    print(f'Starting epoch {epoch+1}')
    start = time.time()
    for i, (inputs, targets) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = net(inputs)
        loss = combined_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        if (i+1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')
        
        # Calculate average loss for the epoch
        avg_loss = loss / len(dataloader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
    
    # Log images
    if epoch % log_image_interval == 0:  # log_image_interval is how often you want to log images
        # Unnormalize and log input images
        input_images = vutils.make_grid(unnormalize(inputs[:n_images].clone().cpu()), nrow=4)  # Log n_images
        writer.add_image('Input images', input_images, global_step=epoch * len(dataloader) + i)

        # Unnormalize and log target images
        target_images = vutils.make_grid(unnormalize(targets[:n_images].clone().cpu()), nrow=4)
        writer.add_image('Target images', target_images, global_step=epoch * len(dataloader) + i)

        # Unnormalize and log output images
        output_images = vutils.make_grid(unnormalize(outputs[:n_images].detach().clone().cpu()), nrow=4)
        writer.add_image('Output images', output_images, global_step=epoch * len(dataloader) + i)

    for name, param in net.named_parameters():
        if param.requires_grad and param.grad is not None:
            writer.add_histogram(f'{name}.grad', param.grad, epoch)

    end = time.time()
    print(f'Finished epoch {epoch+1} in {end-start} seconds')

    # Save checkpoint periodically
    if (epoch+1) % save_checkpoint_every == 0 or epoch == 0:
        checkpoint_filename = f'models/{MODEL_NAME}_epoch_{epoch+1}.pth'
        torch.save({
            'epoch': epoch+1,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }, checkpoint_filename)
        print(f'Saved checkpoint for epoch {epoch+1}')
    
    scheduler.step()
    writer.add_scalar('Learning Rate', scheduler.get_last_lr()[0], epoch)

print('Finished Training')
# Save the final model checkpoint
checkpoint_filename = f'models/{MODEL_NAME}_epoch_{epoch+1}.pth'
torch.save({
    'epoch': epoch+1,
    'model_state_dict': net.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
}, checkpoint_filename)

torch.save(net.state_dict(), f'models/{MODEL_NAME}_final.pth')
