"""Download and verify the FlashRank model without loading ONNX Runtime."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory

import httpx

FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"
FLASHRANK_MODEL_REVISION = "858a1ac046a05663a35367eac852d7f76feeefdd"
FLASHRANK_MODEL_URL = (
    "https://huggingface.co/prithivida/flashrank/resolve/"
    f"{FLASHRANK_MODEL_REVISION}/{FLASHRANK_MODEL}.zip"
)
FLASHRANK_MODEL_SHA256 = "bdd3772b651ffc34f70e414049285bb55ccc6d1b8e29d0640f836d44f70ec77a"
DEFAULT_RERANKER_CACHE_DIR = Path("data/models/flashrank")


def prepare_flashrank_model(cache_dir: Path = DEFAULT_RERANKER_CACHE_DIR) -> None:
    """Download, verify, and install the immutable FlashRank artifact."""
    model_dir = cache_dir / FLASHRANK_MODEL
    if _is_verified_flashrank_model(model_dir):
        return
    if model_dir.exists():
        raise ValueError(f"Unverified FlashRank model directory: {model_dir}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=cache_dir) as staging_dir:
        staging_root = Path(staging_dir)
        archive = staging_root / f"{FLASHRANK_MODEL}.zip"
        with httpx.stream(
            "GET",
            FLASHRANK_MODEL_URL,
            follow_redirects=True,
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            with archive.open("wb") as file:
                for chunk in response.iter_bytes():
                    file.write(chunk)
        if _sha256(archive) != FLASHRANK_MODEL_SHA256:
            raise ValueError("Downloaded FlashRank archive checksum does not match")

        extraction_root = staging_root / "extracted"
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
            allowed_prefixes = (f"{FLASHRANK_MODEL}/", "__MACOSX/")
            if not names or any(
                PurePosixPath(name).is_absolute()
                or ".." in PurePosixPath(name).parts
                or not name.startswith(allowed_prefixes)
                for name in names
            ):
                raise ValueError("Downloaded FlashRank archive has an unexpected layout")
            for name in names:
                if name.startswith(f"{FLASHRANK_MODEL}/"):
                    bundle.extract(name, extraction_root)
        staged_model_dir = extraction_root / FLASHRANK_MODEL
        staged_marker = staged_model_dir / ".artifact-sha256"
        staged_marker.write_text(f"{FLASHRANK_MODEL_SHA256}\n", encoding="utf-8")
        try:
            staged_model_dir.replace(model_dir)
        except OSError:
            if not _is_verified_flashrank_model(model_dir):
                raise


def _is_verified_flashrank_model(model_dir: Path) -> bool:
    checksum_marker = model_dir / ".artifact-sha256"
    return (
        model_dir.is_dir()
        and checksum_marker.is_file()
        and checksum_marker.read_text(encoding="utf-8").strip() == FLASHRANK_MODEL_SHA256
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
