import cv2
import numpy as np

def shift_object(original_path, mask_path, background_path, shift_x, shift_y, output_path):
    # Load images
    original = cv2.imread(original_path)
    mask = cv2.imread(mask_path, 0)
    background = cv2.imread(background_path)
    print("original.shape", original.shape)
    print("mask.shape", mask.shape)
    print("background.shape", background.shape)
    
    # Check if dimensions match
    if original.shape[:2] != mask.shape[:2] or original.shape[:2] != background.shape[:2]:
        raise ValueError("All images must have the same dimensions.")
    
    # Extract the object using the mask
    object = cv2.bitwise_and(original, original, mask=mask)
    
    # Create an empty matrix for shifted object
    shifted_object = np.zeros_like(original)
    
    # Define the shift in x and y directions
    tx = shift_x
    ty = -shift_y  # Negative because positive y means moving down in image coordinates
    
    # Translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted_object = cv2.warpAffine(object, translation_matrix, (shifted_object.shape[1], shifted_object.shape[0]))
    
    # Where the object is placed, we zero out the background
    mask_shifted = cv2.warpAffine(mask, translation_matrix, (mask.shape[1], mask.shape[0]))
    background[mask_shifted > 0] = 0
    
    # Add the shifted object to the background
    final_image = cv2.add(background, shifted_object)
    print("final_image.shape", final_image.shape)
    
    # Save the result
    cv2.imwrite(output_path, final_image)
    print(f"Output saved as {output_path}")

# Example usage:
shift_object('to_delete_stool/original.jpg', 'to_delete_stool/mask.png', 'to_delete_stool/background.jpg', 48, -30, 'to_delete_stool/updated_image_correct_1.jpg')
