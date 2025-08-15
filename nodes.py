from contextlib import contextmanager
import os
from huggingface_hub import hf_hub_download, login, logout


class HFHubModelDownloader:
    """Download models from huggingface hub"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "repo_id": ("STRING", {"default": ""}),
                "filename": ("STRING", {"default": ""}),
                "model_type": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "download_model"

    CATEGORY = "MICO World"

    def download_model(self, repo_id: str, filename: str, model_type: str):
        with hf_login():
            print(f"Download Model: {filename} From HF Hub")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=get_comfy_dir(model_type)
            )
        return filename


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


@contextmanager
def hf_login(token):
    try:
        login(token)
        print(f"huggingface login with {token}")
    finally:
        logout()
        print(f"huggingface logout")

NODE_CLASS_MAPPINGS = {
    "HFHubModelDownloader": HFHubModelDownloader
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HFHubModelDownloader": "HF Hub Model Downloader"
}
