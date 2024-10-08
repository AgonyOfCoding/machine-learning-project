import os
from PIL import Image
import matplotlib.pyplot as plt

def get_first_images_and_labels(root_dir, num_images=10):
    """
    Gets the first image and corresponding class name from each symbol class folder.

    Parameters:
    - root_dir: The path to the directory containing symbol class folders.
    - num_images: The total number of images to retrieve (10 by default).

    Returns:
    - A list of tuples containing (Image object, class name).
    """
    class_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                     if os.path.isdir(os.path.join(root_dir, d))]
    
    selected_images_and_labels = []
    
    for folder in sorted(class_folders)[:num_images]:
        image_files = sorted([f for f in os.listdir(folder) if f.endswith('.png') or f.endswith('.jpg')])
        if image_files:
            first_image_path = os.path.join(folder, image_files[400])
            try:
                image = Image.open(first_image_path)
                class_name = os.path.basename(folder)
                selected_images_and_labels.append((image, class_name))
            except Exception as e:
                print(f"Error: Could not open image '{first_image_path}'. Error: {e}")
                continue
    
    return selected_images_and_labels

def display_images_in_grid_with_labels(images_and_labels, num_rows=2, num_cols=5):
    """
    Displays a list of images with their class names in a grid layout using Matplotlib.

    Parameters:
    - images_and_labels: List of tuples containing (Image object, class name).
    - num_rows: Number of rows in the grid.
    - num_cols: Number of columns in the grid.
    """
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))
    axes = axes.flatten()  # Flatten to 1D array for easier iteration

    for i, ax in enumerate(axes):
        if i < len(images_and_labels):
            image, class_name = images_and_labels[i]
            ax.imshow(image)
            ax.set_title(class_name, fontsize=10)  # Set the class name as the title
            ax.axis('off')  # Hide axes
        else:
            ax.axis('off')  # Hide axes for empty subplots

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set the directory containing the symbol class folders
    root_dir = "./dataset_classification"  # Replace with your actual output directory path
    
    # Get the first image and label from each folder (limit to 10 images for 2 rows of 5)
    first_images_and_labels = get_first_images_and_labels(root_dir, num_images=10)
    
    # Display the images and class names in a grid (2 rows, 5 columns)
    display_images_in_grid_with_labels(first_images_and_labels, num_rows=2, num_cols=5)
