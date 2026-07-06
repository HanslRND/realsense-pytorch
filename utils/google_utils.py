from pathlib import Path


def attempt_download(file, repo=""):
    if not Path(file).exists():
        raise FileNotFoundError(f"Weights not found: {file}")
