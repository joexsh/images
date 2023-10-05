import os
import numpy as np
from skimage import io, filters, morphology, segmentation

# Define the directory containing the TIFF images
input_directory = '/Users/shejoev/Documents/scan/bridge_sample/convertion/L1_copy'

# Define the output directory to save segmented images
output_directory = '/Users/shejoev/Documents/scan/bridge_sample/convertion/L1_segmentation'

# Ensure the output directory exists
os.makedirs(output_directory, exist_ok=True)

# List all TIFF files in the input directory
image_files = [f for f in os.listdir(input_directory) if f.endswith('.tiff')]

# Iterate through each TIFF image
for image_file in image_files:
    # Load the TIFF image
    image_path = os.path.join(input_directory, image_file)
    img = io.imread(image_path)

    # Apply histogram-based thresholding (you may need to adjust the threshold)
    threshold_value = filters.threshold_otsu(img)
    binary_image = img > threshold_value

    # Perform morphological operations to clean up the binary image
    cleaned_image = morphology.remove_small_objects(binary_image, min_size=100)

    # Label connected components (objects)
    labeled_image, num_objects = morphology.label(cleaned_image, connectivity=2, return_num=True)

    # Create an array to store the segmented image
    segmented_image = np.zeros_like(img)

    # Set all background pixels to zero
    for label in range(1, num_objects + 1):
        segmented_image[labeled_image == label] = img[labeled_image == label]

    # Save the segmented image
    output_path = os.path.join(output_directory, f'segmented_{image_file}')
    io.imsave(output_path, segmented_image)

    print(f'Segmented and saved: {image_file}')
