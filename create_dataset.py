import os
import glob
import random
from PIL import Image

def find_images(root_dir):
    """Recursively find all jpg and png files."""
    return glob.glob(root_dir + '/**/*.jpg', recursive=True) + glob.glob(root_dir + '/**/*.png', recursive=True)

def create_dataset(root_dir, output_dir, image_size=256, expansion_factor=1.10, samples_per_image=5):
    images = find_images(root_dir)
    os.makedirs(os.path.join(output_dir, 'input'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'output'), exist_ok=True)

    count = 0
    for img_path in images:
        try:
            
            if os.path.getsize(img_path) > 10*1024*1024:
                print(f"Skipping large file: {img_path}")
                continue
            
            with Image.open(img_path) as img:
                # Convert image to RGB if it's RGBA or P (Palette)
                if img.mode in ('RGBA', 'P'):
                    img = img.convert('RGB')
                
                

                # Scale down large images
                scale_factor = min(img.width / image_size, img.height / image_size)
                if scale_factor > 1.5:  # Adjust this factor as needed
                    scale_factor = scale_factor / random.uniform(1.5, 3.0)
                    new_size = (int(img.width / scale_factor), int(img.height / scale_factor))
                    img = img.resize(new_size, Image.LANCZOS)

                for _ in range(samples_per_image):
                    if img.width > image_size and img.height > image_size:
                        print("processing image: " + img_path)
                        x = random.randint(0, img.width - image_size)
                        y = random.randint(0, img.height - image_size)
                        input_crop = img.crop((x, y, x + image_size, y + image_size))

                        # Calculate the center of the original crop
                        center_x, center_y = x + image_size // 2, y + image_size // 2

                        # Calculate new top-left and bottom-right coordinates for expanded crop
                        expanded_half_width = int(image_size * expansion_factor) // 2
                        expanded_half_height = int(image_size * expansion_factor) // 2

                        new_x1 = max(0, center_x - expanded_half_width)
                        new_y1 = max(0, center_y - expanded_half_height)
                        new_x2 = min(img.width, center_x + expanded_half_width)
                        new_y2 = min(img.height, center_y + expanded_half_height)

                        expanded_crop = img.crop((new_x1, new_y1, new_x2, new_y2)).resize((image_size, image_size), Image.LANCZOS)

                        # Save the crops
                        print("saving image: " + str(count) + ".png")
                        input_crop.save(os.path.join(output_dir, 'input', f'{count:04}.jpg'))
                        expanded_crop.save(os.path.join(output_dir, 'output', f'{count:04}.jpg'))
                        count += 1
                    else:
                        print("skipping image: " + img_path, img.width, img.height, image_size)

        except:
            print("Error processing image: " + img_path)
        if count > 5000:
            break

# Usage
create_dataset('/media/ccc/scroopynoop/bad/images', 'dataset')
