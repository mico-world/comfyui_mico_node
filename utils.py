from contextlib import contextmanager

from huggingface_hub import hf_hub_download, login, logout

class HFUtils:

    def __init__(self, token):
        self.token = token

    def download(self, repo_id: str, filename: str, local_dir: str):
        with HFUtils.hf_login(self.token):
            print(f"Download Model: {filename} From HF Hub")
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=local_dir
            )

    @contextmanager
    @staticmethod
    def hf_login(token):
        try:
            login(token)
            print(f"huggingface login with {token}")
        finally:
            logout()
            print(f"huggingface logout")


