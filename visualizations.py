import os
import matplotlib.pyplot as plt
import numpy as np

# Define the directory path
directory_path = 'images/Images'

# Get the absolute path of the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Construct the full path
full_directory_path = os.path.join(current_directory, directory_path)

# Initialize an empty dictionary to store folder names and their corresponding image counts
folder_image_counts = {}

# Iterate over each folder in the directory
for folder_name in os.listdir(full_directory_path):
    folder_path = os.path.join(full_directory_path, folder_name)
    
    # Check if the path is a directory
    if os.path.isdir(folder_path):
        # Extract the name after the "-" in the folder name
        display_name = folder_name.split('-')[-1].strip()
        
        # Count the number of files (images) in the folder
        num_images = len([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
        
        # Store the display name and its image count in the dictionary
        folder_image_counts[display_name] = num_images

# Sort the dictionary by values
sorted_counts = sorted(folder_image_counts.items(), key=lambda x: x[1])

# Get the 5 categories with the largest counts
top_3_largest = sorted_counts[-3:]

# Get the 5 categories with the smallest counts
top_3_smallest = sorted_counts[:3]

# Calculate the median count
median_count = np.median(list(folder_image_counts.values()))

# Get the 5 categories that are closest to the median count
closest_to_median = sorted(sorted_counts, key=lambda x: abs(x[1] - median_count))[:3]

# Combine the selected categories
selected_categories = top_3_largest + top_3_smallest + closest_to_median

# Create a horizontal bar chart with a nicer color
plt.figure(figsize=(12, 10))
bars = plt.barh([item[0] for item in selected_categories], [item[1] for item in selected_categories], color='skyblue')
plt.xlabel('Number of Images')
plt.ylabel('Dog Breeds')
plt.title('Number of Images for Selected Dog Breeds')
plt.grid(axis='x')  # Display grid lines on the x-axis
plt.tight_layout()

# Add values on top of the bars
for bar in bars:
    plt.text(bar.get_width() - 5, bar.get_y() + bar.get_height() / 2 - 0.1, str(int(bar.get_width())), 
             va='center', ha='right', color='black', fontsize=8, fontweight='bold')

# Save the plot as a PNG file
plt.savefig('selected_dog_breeds.png', dpi=300, bbox_inches='tight')

plt.show()
