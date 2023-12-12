from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class ImagePairDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(os.path.join(root_dir, 'input'))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        input_path = os.path.join(self.root_dir, 'input', self.image_files[idx])
        output_path = os.path.join(self.root_dir, 'output', self.image_files[idx])

        input_image = Image.open(input_path).convert('RGB')
        output_image = Image.open(output_path).convert('RGB')

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image

# # Example of how to use the dataset
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# dataset = ImagePairDataset(root_dir='/path/to/dataset', transform=transform)
# dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # Example of iterating over the dataset
# for input_image, output_image in dataloader:
#     # Perform operations with input_image and output_image
