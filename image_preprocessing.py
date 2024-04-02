import numpy as np
import cv2
import os

    
def crop_image(image, crop_size):
    # Crop image to specified size
    h, w = image.shape[:2]
    if h > crop_size[0] and w > crop_size[1]:
        return image[(h - crop_size[0]) // 2:(h + crop_size[0]) // 2, (w - crop_size[1]) // 2:(w + crop_size[1]) // 2]
    else:
        return cv2.resize(image, crop_size)
    

def vectorize_image(image):
    # Convert image to 1d vector (np array)
    return image.flatten()


def preprocessing(input_dirs, output_parent_dir, crop_size=(300, 300)):
    # Processes images from input directory, outputs to output directory
    for sub_dir in input_dirs: # subdir = 1 breed
        print(f"Processing images in directory: {sub_dir}")
        breed_name = os.path.basename(sub_dir)  # Get the name of the breed folder
        output_dir = os.path.join(output_parent_dir, breed_name)  # Create output directory for the breed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for filename in os.listdir(sub_dir):
            if filename.endswith('.jpg'):
                image_path = os.path.join(sub_dir, filename)
                image = cv2.imread(image_path)
                if image is not None:
                    cropped_image = crop_image(image, crop_size)
                    vectorized_image = vectorize_image(cropped_image)
                    output_path = os.path.join(output_dir, filename)
                    np.save(output_path, vectorized_image)


# Directory containing folders for different dog breeds
input_parent_dir = r"C:\Users\sophi\Dog-Breed-Classification\images\Images"

# List of input directories (folders for different dog breeds)
input_dirs = [os.path.join(input_parent_dir, folder) for folder in os.listdir(input_parent_dir)]

# Output directory where processed images will be saved
output_parent_dir = "preprocessed_images"

# Preprocess images
preprocessing(input_dirs, output_parent_dir)
