import os
import csv
from PIL import Image
import numpy as np

# Define the directory containing the segmented TIFF images
input_directory = '/Users/shejoev/Documents/scan/CT-1/convex_true/log1-1'

# List all TIFF files in the input directory
image_files = [f for f in os.listdir(input_directory) if f.endswith('.tif')]

# Create a CSV file to store the pixel count data
output_csv_path = '/Users/shejoev/Documents/scan/CT-1/convex_true/data_frame/log_1_1/data.csv'
with open(output_csv_path, mode='w', newline='') as csv_file:
    fieldnames = ['ImageName', 'Value1', 'Value2', 'Value3', 'Value0', 'TotalPixels']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
    # Write the header row to the CSV file
    writer.writeheader()
    
    # Loop through each segmented image
    for image_file in image_files:
        # Load the segmented image using Pillow
        image_path = os.path.join(input_directory, image_file)
        segmented_image = Image.open(image_path)
        
        # Convert the image to a NumPy array
        segmented_image_array = np.array(segmented_image)
        
        # Count the pixels for each value (0, 1, 2, 3)
        pixel_counts = [np.count_nonzero(segmented_image_array == value) for value in range(4)]
        
        # Calculate the total number of pixels
        total_pixels = segmented_image_array.size
        
        # Write a row to the CSV file
        writer.writerow({
            'ImageName': image_file,
            'Value1': pixel_counts[1],
            'Value2': pixel_counts[2],
            'Value3': pixel_counts[3],
            'Value0': pixel_counts[0],
            'TotalPixels': total_pixels
        })
    
# Display the CSV file path
print(f'CSV file saved at: {output_csv_path}')
