import os
import json
from PIL import Image
import shutil
from tqdm import tqdm

included_classes = [
    "ProcessInstrumentationFunction",
    "Flange",
    "GateValve",
    "BallValve",
    "OPC",
    "ActuatingSystemComponent",
    "PipeFitting",
    "ConcentricDiameterChange",
    "Actuator",
    "Tank"
]

def resize_and_center_symbol(cropped_image, target_size=128, symbol_max_size=123):
    """
    Resizes the cropped symbol to fit within a target size while maintaining aspect ratio.
    The larger side of the symbol is resized to symbol_max_size pixels.
    The symbol is then centered on a white background of target_size x target_size.
    """
    # Determine the scaling factor
    width, height = cropped_image.size
    if width > height:
        new_width = symbol_max_size
        new_height = int((symbol_max_size / width) * height)
    else:
        new_height = symbol_max_size
        new_width = int((symbol_max_size / height) * width)
    
    # Resize the symbol
    resized_symbol = cropped_image.resize((new_width, new_height))
    
    # Create a white background
    new_image = Image.new("RGBA", (target_size, target_size), (255, 255, 255, 0))
    
    # Calculate positioning to center the symbol
    paste_x = (target_size - new_width) // 2
    paste_y = (target_size - new_height) // 2
    
    # Paste the resized symbol onto the background
    new_image.paste(resized_symbol, (paste_x, paste_y), resized_symbol.convert("RGBA"))
    
    # Convert to RGB (optional, depending on your needs)
    final_image = new_image.convert("RGB")
    
    return final_image

def sanitize_class_name(class_name):
    """
    Sanitizes the class name to be a valid folder name.
    Removes or replaces characters that are invalid in folder names.
    """
    valid_name = "".join(c if c.isalnum() or c in (' ', '_') else '_' for c in class_name)
    valid_name = valid_name.replace(' ', '_')
    return valid_name

def preprocess_dataset(root_dir, output_dir):
    """
    Preprocesses the dataset by cropping, resizing, and organizing symbols into class-specific folders.
    
    Parameters:
    - root_dir: Path to the root directory containing project folders.
    - output_dir: Path to the directory where processed images will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize a dictionary to keep track of image counts per class
    class_image_count = {}
    
    # Iterate through each project folder
    project_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]
    
    for project in tqdm(project_folders, desc="Processing Projects"):
        # Locate the JSON annotation file (assuming only one per project)
        json_files = [f for f in os.listdir(project) if f.endswith('.json')]
        if not json_files:
            print(f"Warning: No JSON file found in project folder '{project}'. Skipping.")
            continue
        json_path = os.path.join(project, json_files[0])
        
        # Load JSON data
        with open(json_path, 'r') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error: Failed to parse JSON file '{json_path}'. Skipping.")
                continue
        
        # Iterate through each page in the JSON data
        pages = data.get("pages", [])
        for page in pages:
            image_path = page.get("imagePath", "")
            image_file = os.path.join(project, image_path)
            
            # Skip missing images
            if not os.path.exists(image_file):
                print(f"Warning: Image file '{image_file}' does not exist. Skipping.")
                continue
            
            try:
                with Image.open(image_file) as img:
                    img = img.convert("RGBA")  # Ensure image has an alpha channel for transparency
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error: Failed to open image '{image_file}'. Error: {e}. Skipping.")
                continue
            
            # Process each label (symbol) in the selected image
            labels = page.get("labels", [])
            for label in labels:
                class_name = label.get("className", "unknown")
                if (not (class_name in included_classes)):
                    continue
                bbox = label.get("bbox", [])
                if not bbox or len(bbox) != 4:
                    print(f"Warning: Invalid bbox for label in image '{image_file}'. Skipping label.")
                    continue
                
                x, y, width, height = bbox
                # Ensure bbox is within image boundaries
                x = max(0, x)
                y = max(0, y)
                width = min(width, img_width - x)
                height = min(height, img_height - y)
                
                # Crop the symbol from the image
                cropped = img.crop((x, y, x + width, y + height))
                
                # Resize and center the symbol
                processed_symbol = resize_and_center_symbol(cropped, target_size=64, symbol_max_size=59)
                
                # Sanitize class name for folder naming
                sanitized_class = sanitize_class_name(class_name)
                class_folder = os.path.join(output_dir, sanitized_class)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                
                # Initialize or increment the image count for naming
                if sanitized_class not in class_image_count:
                    class_image_count[sanitized_class] = 1
                else:
                    class_image_count[sanitized_class] += 1
                image_filename = f"image{class_image_count[sanitized_class]}.png"
                
                # Save the processed symbol image
                save_path = os.path.join(class_folder, image_filename)
                try:
                    processed_symbol.save(save_path, format='PNG')
                except Exception as e:
                    print(f"Error: Failed to save image '{save_path}'. Error: {e}. Skipping.")
                    continue

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    # Define the root directory containing project folders
    ROOT_DIR = "./dataset_original"  # Replace with your actual root directory path
    
    # Define the output directory where processed images will be saved
    OUTPUT_DIR = "./dataset_classification"  # Replace with your desired output directory path
    
    preprocess_dataset(ROOT_DIR, OUTPUT_DIR)
