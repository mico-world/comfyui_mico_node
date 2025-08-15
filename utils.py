from contextlib import contextmanager
import os

from huggingface_hub import hf_hub_download, login, logout

DATA_UNITS = ('B', 'KB', 'MB', 'GB')

class HFUtils:

    def __init__(self, token: str=""):
        self.token = token

    @contextmanager
    def login(self):
        try:
            if self.token:
                print(f'HF login use: {self.token}')
                login(self.token)
            yield True
        finally:
            logout()
            print(f"huggingface logout")
        

    def download(self, repo_id: str, filename: str, local_dir: str, data_units=2):
        with self.login():
            print(f"Download Model: {filename} From HF Hub")
            path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir
            )
            file_size_bytes = os.path.getsize(os.path.join(path))
            for i in range(data_units):
                file_size_bytes /= 1024
            print(f"✅ diffusion model {filename} download success, size: {file_size_bytes:.2f}{DATA_UNITS[data_units]}")
            



