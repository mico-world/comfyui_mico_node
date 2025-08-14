from io import BytesIO
import httpx
import numpy as np
import torch
from PIL import Image, ImageSequence, ImageOps
from server import PromptServer

class LoadImageFromUrl:
    CATEGORY = "Comfyui_MICO"
    @classmethod    
    def INPUT_TYPES(s):
        return { 
            "required":  { 
                "image": (
                    "STRING", 
                    {
                        "multiline": False,
                        "default": "https://kivi-ai.micoplatform.com/web/static/default_1.1e72e060.png"
                    }),
            } 
        }
    
    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "load_image"

    def load_image(self, image: str):
        if not image.startswith('http'):
            raise ValueError('LoadImageFromUrl input should be url')
    
        resp = httpx.get(image, timeout=60.0)
        image_path = BytesIO(resp.content)
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        w, h = None, None
        excluded_formats = ['MPO']

        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")

            if len(output_images) == 0:
                w = image.size[0]
                h = image.size[1]

            if image.size[0] != w or image.size[1] != h:
                continue

            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            elif i.mode == 'P' and 'transparency' in i.info:
                mask = np.array(i.convert('RGBA').getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))
        
        if len(output_images) > 1 and img.format not in excluded_formats:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]

        return (output_image, output_mask)
        


        
           

            

        
