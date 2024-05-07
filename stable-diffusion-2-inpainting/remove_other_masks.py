from PIL import Image
import numpy as np

# Load the image
image_path = 'temp_outputs/wall_hanging_multiple_masks.png'
img = Image.open(image_path)
img_array = np.array(img)

# Get unique values in the array
unique_values = np.unique(img_array)
print(unique_values)

# Choose the value to isolate (using 100 as an example)
selected_value = 228

# Create a mask where only the pixels with the selected value are retained
selected_mask = (img_array == selected_value).astype(np.uint8) * 255  # Multiplying by 255 to make the selected area visible

# Save the masked image as a PNG
output_path = 'temp_outputs/mask_228.png'
Image.fromarray(selected_mask).save(output_path)

output_path