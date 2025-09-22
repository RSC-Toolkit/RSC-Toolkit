import os
import requests
import tqdm
import urllib3
import logging
import zipfile


logger = logging.getLogger(__name__)


def download_file(url: str, out_path: str, chunk_size: int = 1024):
    """Download file with progress bar, retry without cert verification if SSL fails."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    try:
        with requests.get(url, stream=True, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(out_path, "wb") as f, tqdm.tqdm(
                total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
    except requests.exceptions.SSLError:
        logger.warning("⚠️ SSL error, retrying without certificate verification...")
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        with requests.get(url, stream=True, verify=False, timeout=30) as r:
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(out_path, "wb") as f, tqdm.tqdm(
                total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)
            ) as pbar:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))


def extract_zip(zip_path: str, out_dir: str):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)


