"""Runtime preparation utilities for Vertex training jobs."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import login

from .utils import configure_logging

logger = configure_logging()


def login_hf(token_env: str = "HF_TOKEN") -> None:
    """Authenticate with Hugging Face Hub using a token from the environment."""

    token = os.environ.get(token_env)
    if not token:
        logger.warning("HF token environment variable %s not set; skipping login", token_env)
        return
    login(token=token)
    logger.info("Authenticated with Hugging Face Hub using token env %s", token_env)


def install_flash_attn_from_gcs(gcs_wheel_uri: str, target_dir: str = "/tmp/wheels") -> bool:
    """Download a FlashAttention wheel from GCS and install it without deps."""

    if not gcs_wheel_uri:
        logger.warning("No FlashAttention wheel URI provided; skipping installation")
        return False
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    local_path = Path(target_dir) / Path(gcs_wheel_uri).name
    cmd = ["gcloud", "storage", "cp", gcs_wheel_uri, str(local_path)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("Failed to download FlashAttention wheel: %s", result.stderr)
        return False
    install_cmd = ["pip", "install", "--no-deps", str(local_path)]
    install_result = subprocess.run(install_cmd, capture_output=True, text=True, check=False)
    if install_result.returncode == 0:
        logger.info("Installed FlashAttention wheel from %s", gcs_wheel_uri)
        return True
    logger.error("Failed to install FlashAttention: %s", install_result.stderr)
    return False


def enable_flash_attn_if_available(*, log: bool = True) -> bool:
    """Enable PyTorch scaled-dot-product attention optimizations when available."""

    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        if log:
            logger.info("Enabled FlashAttention/SDP kernels")
        else:
            logger.debug("FlashAttention/SDP kernels enabled")
        return True
    except Exception as exc:  # pragma: no cover - safety net
        logger.warning("Could not enable FlashAttention kernels: %s", exc)
        return False
