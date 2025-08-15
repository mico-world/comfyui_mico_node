import os
import torch
import folder_paths
import comfy.sd
from ..utils import HFUtils


class HFUNETLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": ""}),
                "weight_dtype": (["default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2"],),
                "api_key": ("STRING", {"default": ""}),
            }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"

    CATEGORY = "MICO World/loaders"

    def load_unet(self, repo_id: str, filename: str, weight_dtype: str, api_key: str):
        model_options = {}
        if weight_dtype == "fp8_e4m3fn":
            model_options["dtype"] = torch.float8_e4m3fn
        elif weight_dtype == "fp8_e4m3fn_fast":
            model_options["dtype"] = torch.float8_e4m3fn
            model_options["fp8_optimizations"] = True
        elif weight_dtype == "fp8_e5m2":
            model_options["dtype"] = torch.float8_e5m2
        try:
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", filename)
        except FileNotFoundError:
            HFUtils(api_key).download(repo_id, filename, os.path.dirname(unet_path))
                
            unet_path = folder_paths.get_full_path_or_raise("diffusion_models", filename)
            
        
        model = comfy.sd.load_diffusion_model(
            unet_path, model_options=model_options)
        return (model,)



def get_comfy_dir(model_type):
    cwd = os.getcwd()
    if os.path.exists(os.path.join(cwd, "ComfyUI")):
        cwd = os.path.join(cwd, "ComfyUI")
    if os.path.exists(os.path.join(cwd, "models")):
        cwd = os.path.join(cwd, "models")

    result = os.path.join(cwd, model_type)

    if os.path.exists(result):
        return result
    raise ValueError(f"Path {result} does not exist!")






