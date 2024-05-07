import os
import sys
import importlib.util
import subprocess
from utils import maskout_segmented_obj
import argparse



grounding_sam_directory = './Grounded_Segment_Anything_main'
stable_diffusion_directory = './stable-diffusion-2-inpainting'



parser = argparse.ArgumentParser(description='Shift prescribed object in the image by x and y pixels')
parser.add_argument('--image_name', type=str, help='Name of the image with extension, e.g., chair.jpg')
parser.add_argument('--class_name', type=str, help='Class/Object name, e.g., chair')
parser.add_argument('--x_distance', type=int, default=1, help='pixels to shift in x direction')
parser.add_argument('--y_distance', type=int, default=1, help='pixels to shift in y direction')

# Parse arguments
args = parser.parse_args()

# Assign arguments to variables
image_name = args.image_name
x_distance = args.y_distance
y_distance = args.x_distance
class_name = args.class_name



################################ Grounding SAM ################################
print('##################################')
print('Current working directory:', os.getcwd())
original_directory = os.getcwd()
os.chdir(grounding_sam_directory)
print('Changed directory to:', os.getcwd())
print('##################################')


command = [
    'python', 'grounded_sam_demo.py',
    '--config', 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
    '--grounded_checkpoint', 'groundingdino_swint_ogc.pth',
    '--sam_checkpoint', 'sam_vit_h_4b8939.pth',
    '--input_image', image_name,
    '--output_dir', './outputs/grounding_sam_output',
    '--box_threshold', '0.3',
    '--text_threshold', '0.25',
    '--text_prompt', class_name,
    '--device', 'cpu'
]
result = subprocess.run(command)

print('##################################')

if result.returncode != 0:
    print(f"Command failed with return code {result.returncode}")
else:
    print("Command executed successfully")

print('Captured the POI object from the image using Grounding SAM')
print('##################################')



################################ Stable Diffusion Inpainting ################################

print('Current working directory:', os.getcwd())
os.chdir(stable_diffusion_directory)
print('Changed directory to:', os.getcwd())

command = [
        'python', 'run.py'
]
result = subprocess.run(command)
if result.returncode != 0:
    print(f"Command failed with return code {result.returncode}")
else:
    print("Command executed successfully")
    


################################ Grounding SAM ################################
print('##################################')
print('Current working directory:', os.getcwd())
original_directory = os.getcwd()
os.chdir(grounding_sam_directory)
print('Changed directory to:', os.getcwd())
print('##################################')


command = [
    'python', 'grounded_sam_demo.py',
    '--config', 'GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py',
    '--grounded_checkpoint', 'groundingdino_swint_ogc.pth',
    '--sam_checkpoint', 'sam_vit_h_4b8939.pth',
    '--input_image', './outputs/zero123_output/rotated_obj.png',
    '--output_dir', './outputs/grounding_sam_output_2',
    '--box_threshold', '0.3',
    '--text_threshold', '0.25',
    '--text_prompt', class_name,
    '--device', 'cpu'
]
result = subprocess.run(command)

print('##################################')

if result.returncode != 0:
    print(f"Command failed with return code {result.returncode}")
else:
    print("Command executed successfully")

print('Captured the POI object from the image using Grounding SAM')
print('##################################')    

################################ Post-Processing ################################

print('Current working directory:', os.getcwd())
os.chdir(original_directory)
print('Changed directory to:', os.getcwd())

command = [
        'python', 'shift_object.py'
]
result = subprocess.run(command)
if result.returncode != 0:
    print(f"Command failed with return code {result.returncode}")
else:
    print("Command executed successfully")