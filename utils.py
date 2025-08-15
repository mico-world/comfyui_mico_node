from contextlib import contextmanager

from huggingface_hub import hf_hub_download, login, logout

class HFUtils:

    def __init__(self, token: str=""):
        self.token = token

    @contextmanager
    def download(self, repo_id: str, filename: str, local_dir: str):
        try:
            if self.token:
                login(self.token)
            print(f"Download Model: {filename} From HF Hub")
            path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    local_dir=local_dir
            )
            yield path
        finally:
            logout()
            print(f"huggingface logout")
