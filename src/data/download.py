# Adapted from https://github.com/pytorch/vision/blob/main/torchvision/datasets/utils.py
import hashlib
from logging import getLogger
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from tqdm import tqdm

_logger = getLogger(__name__)

USER_AGENT = "gmlg/sstnp"


def _urlretrieve(url: str, filename: str | Path, chunk_size: int = 1024 * 32) -> None:
    with urlopen(Request(url, headers={"User-Agent": USER_AGENT})) as response:
        with open(filename, "wb") as fh, tqdm(total=response.length) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))


def calculate_md5(fpath: str | Path, chunk_size: int = 1024 * 1024) -> str:
    md5 = hashlib.md5(usedforsecurity=False)
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: str | Path, md5: str, **kwargs) -> bool:
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: str | Path, md5: str | None = None) -> bool:
    fpath = Path(fpath)
    if not fpath.is_file():
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def _get_redirect_url(url: str, max_hops: int = 3) -> str:
    initial_url = url
    headers = {"Method": "HEAD", "User-Agent": USER_AGENT}
    for _ in range(max_hops + 1):
        with urlopen(Request(url, headers=headers)) as response:
            if response.url == url or response.url is None:
                return url
            url = response.url
    else:
        raise RecursionError(
            f"Request to {initial_url} exceeded {max_hops} redirects. "
            f"The last redirect points to {url}."
        )


def download_url(
    url: str,
    root: str | Path,
    filename: str | Path | None = None,
    md5: str | None = None,
    max_redirect_hops: int = 3,
) -> None:
    """Download a file from url and place it in root.

    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under.
            If None, use the basename of the URL
        md5 (str, optional): MD5 checksum of the download. If None, do not check
        max_redirect_hops (int, optional): Maximum number of redirect hops allowed
    """
    root = Path(root).expanduser()
    if not filename:
        filename = Path(url).name
    fpath = root / filename
    root.mkdir(exist_ok=True, parents=True)

    # check if file is already present locally
    if check_integrity(fpath, md5):
        _logger.info(f"Using downloaded and verified file: {fpath}")
        return

    url = _get_redirect_url(url, max_hops=max_redirect_hops)
    try:
        _logger.info(f"Downloading {url} to {fpath}")
        _urlretrieve(url, fpath)
    except (URLError, OSError) as e:
        if url[:5] == "https":
            url = url.replace("https:", "http:")
            _logger.warning(
                f"Failed download. "
                f"Trying http instead of https. Downloading {url} to {fpath}"
            )
            _urlretrieve(url, fpath)
        else:
            raise e
    if not check_integrity(fpath, md5):
        raise RuntimeError("File not found or corrupted.")
