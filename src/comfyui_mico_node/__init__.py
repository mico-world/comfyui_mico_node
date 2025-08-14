from .nodes import LoadImageFromUrl

NODE_CLASS_MAPPINGS = {
    "LoadImageFromUrl" : LoadImageFromUrl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromUrl": "Load Image from Url",
}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]