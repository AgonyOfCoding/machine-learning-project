import os
import json
from PIL import Image
from tqdm import tqdm

# These are the symbol classes included in the project
included_classes = [
    "ProcessInstrumentationFunction",
    "Flange",
    "GateValve",
    "BallValve",
    "OPC",
    "Actuator",
    "PipeFitting",
    "ConcentricDiameterChange",
    "OperatedValve",
    "Tank"
]

# This function merges some similar symbol classes and renames custom classes to non-custom classes
def curated_class_name(className):
    prefix = "Custom"
    if className.startswith(prefix):
        return className[len(prefix):]
    match className:
        case "ActuatingSystemComponent":
            return "Actuator"
        case "BlindFlange":
            return "Flange"
        case "ClampedFlangeCoupling":
            return "Flange"
        case "ControlledActuator":
            return "Actuator"
        case "Controlvalvesandregulators":
            return "BallValve"
        case "FlangedConnection":
            return "Flange"
        case "FlowInPipeOffPageConnector":
            return "OPC"
        case "FlowInSignalOffPageConnector":
            return "OPC"
        case "FlowOutPipeOffPageConnector":
            return "OPC"
        case "GlobeValve":
            return "BallValve"
        case "Off-lineinstruments":
            return "ProcessInstrumentationFunction"
        case "PipeReducer":
            return "ConcentricDiameterChange"
        case "Systemfunctions":
            return "ProcessInstrumentationFunction"
        case _:
            return className
        
def resize_and_center_symbol(cropped_image, target_size=128, symbol_max_size=123):
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
    
    return new_image

def preprocess_dataset(root_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Initialize a dictionary to keep track of image counts per class
    class_image_count = {}
    
    # Iterate through each project folder
    project_folders = [os.path.join(root_dir, d) for d in os.listdir(root_dir) 
                       if os.path.isdir(os.path.join(root_dir, d))]
    
    for project in tqdm(project_folders, desc="Processing Projects"):
        # Locate the JSON annotation file
        json_file = [f for f in os.listdir(project) if f.endswith('.json')]
        if not json_file:
            print(f"Warning: No JSON file found in project folder '{project}'. Skipping.")
            continue
        json_path = os.path.join(project, json_file[0])
        
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
                continue
            
            try:
                with Image.open(image_file) as img:
                    img = img.convert("RGBA")
                    img_width, img_height = img.size
            except Exception as e:
                print(f"Error: Failed to open image '{image_file}'. Error: {e}. Skipping.")
                continue
            
            # Process each label in the selected image
            labels = page.get("labels", [])
            for label in labels:
                class_name = label.get("className", "unknown")
                class_name = curated_class_name(class_name)
                if (not (class_name in included_classes)):
                    continue
                bbox = label.get("bbox", [])
                if not bbox or len(bbox) != 4:
                    print(f"Warning: Invalid bbox for label in image '{image_file}'. Skipping label.")
                    continue
                
                x1, y1, x2, y2 = bbox
                width = x2 - x1
                height = y2 - y1

                # Ensure bbox is within image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                width = min(width, img_width - x1)
                height = min(height, img_height - y1)

                # Skip bad bboxes
                if (x1 > x2 or y1 > y2 or width <= 0 or height <= 0):
                    print(f"Warning: Invalid bbox coordinates for label in image '{image_file}'. Skipping label.")
                    continue
                
                # Crop the symbol from the image
                cropped = img.crop((x1, y1, x2, y2))
                
                # Resize and center the symbol
                processed_symbol = resize_and_center_symbol(cropped, target_size=128, symbol_max_size=123)
                
                # Sanitize class name for folder naming
                class_folder = os.path.join(output_dir, class_name)
                if not os.path.exists(class_folder):
                    os.makedirs(class_folder)
                
                # Initialize or increment the image count for naming
                if class_name not in class_image_count:
                    class_image_count[class_name] = 1
                else:
                    class_image_count[class_name] += 1
                image_filename = f"image{class_image_count[class_name]}.png"
                
                # Save the processed symbol image
                save_path = os.path.join(class_folder, image_filename)
                try:
                    processed_symbol.save(save_path, format='PNG')
                except Exception as e:
                    print(f"Error: Failed to save image '{save_path}'. Error: {e}. Skipping.")
                    continue

    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    ROOT_DIR = "./dataset_original"
    OUTPUT_DIR = "./dataset_classification"    
    preprocess_dataset(ROOT_DIR, OUTPUT_DIR)
