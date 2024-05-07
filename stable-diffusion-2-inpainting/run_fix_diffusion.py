from diffusers import StableDiffusionInpaintPipeline
from diffusers import StableDiffusionImageVariationPipeline
from PIL import Image
import torch
import numpy as np
import cv2


image = Image.open("to_delete_bagpack/updated_image_wrong.jpg")
# mask_image = Image.open('temp_outputs/wall_hanging_mask.png')

#get size of mask_image
width, height = image.size



prompt = "the bag is incomplete, complete the remaining part of the bag"





pipe = StableDiffusionImageVariationPipeline.from_pretrained(
    "lambdalabs/sd-image-variations-diffusers", revision="v2.0"
)
pipe = pipe.to("cuda")
out = pipe(image, num_images_per_prompt=5, guidance_scale=15)


# out = image.resize((width, height))

image_0 = out["images"][0].resize((width, height))
image_0.save("to_delete_bagpack/wrong_image_through_diffusion_0.jpg")
image_1 = out["images"][1].resize((width, height))
image_1.save("to_delete_bagpack/wrong_image_through_diffusion_1.jpg")
image_2 = out["images"][2].resize((width, height))
image_2.save("to_delete_bagpack/wrong_image_through_diffusion_2.jpg")
image_3 = out["images"][3].resize((width, height))
image_3.save("to_delete_bagpack/wrong_image_through_diffusion_3.jpg")
image_4 = out["images"][4].resize((width, height))
image_4.save("to_delete_bagpack/wrong_image_through_diffusion_4.jpg")
# out["images"][1].save("to_delete_bagpack/wrong_image_through_diffusion_1.jpg")
# out["images"][2].save("to_delete_bagpack/wrong_image_through_diffusion_2.jpg")
# out["images"][3].save("to_delete_bagpack/wrong_image_through_diffusion_3.jpg")
# out["images"][4].save("to_delete_bagpack/wrong_image_through_diffusion_4.jpg")
