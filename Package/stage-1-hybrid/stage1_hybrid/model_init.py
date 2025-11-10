"""Student model/tokenizer initialization (HF folder; no runtime vocab resizing)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from .runtime_setup import enable_flash_attn_if_available
from .utils import configure_logging, ensure_dir

logger = configure_logging()

# Which blocks (if any) your training loop treats as "trainable" for provenance.
_FROZEN_BLOCK_NAMES = ("block_0", "block_1")


# -------------------------
# HF loaders (preferred)
# -------------------------
def _embedding_rows(model: nn.Module) -> int:
    if hasattr(model, "get_input_embeddings"):
        emb = model.get_input_embeddings()
        if hasattr(emb, "num_embeddings"):
            return int(emb.num_embeddings)
    # Fallback: common attribute name
    if hasattr(model, "embed") and hasattr(model.embed, "num_embeddings"):
        return int(model.embed.num_embeddings)
    return -1


def _retie_if_possible(model: nn.Module) -> None:
    # Some HF models expose tie_weights(); safe to call if present.
    if hasattr(model, "tie_weights"):
        try:
            model.tie_weights()
        except Exception as e:
            logger.warning("tie_weights() failed (continuing): %s", e)


def _sanity_readout(model: nn.Module, tok) -> Tuple[int, int, int, int]:
    len_tok = int(len(tok))
    tok_vocab_size = int(getattr(tok, "vocab_size", len_tok))
    cfg_vocab = int(getattr(getattr(model, "config", object()), "vocab_size", -1))
    emb_rows = _embedding_rows(model)
    logger.info(
        "Sanity readout — len(tokenizer)=%d | tokenizer.vocab_size=%d | config.vocab_size=%d | emb_rows=%d",
        len_tok, tok_vocab_size, cfg_vocab, emb_rows,
    )
    return len_tok, tok_vocab_size, cfg_vocab, emb_rows


def _check_vocab(
    model: nn.Module,
    tok,
    expected_vocab_size: Optional[int] = None,
    strict: bool = False,
) -> None:
    emb_rows = _embedding_rows(model)
    cfg_vocab = int(getattr(getattr(model, "config", object()), "vocab_size", -1))
    len_tok = int(len(tok))

    # Basic internal consistency
    if emb_rows != cfg_vocab:
        msg = f"Model mismatch: embeddings={emb_rows} vs config.vocab_size={cfg_vocab}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    if cfg_vocab != len_tok:
        msg = f"Tokenizer/model mismatch: len(tokenizer)={len_tok} vs config.vocab_size={cfg_vocab}"
        if strict:
            raise ValueError(msg)
        logger.warning(msg)

    if expected_vocab_size is not None and cfg_vocab != expected_vocab_size:
        msg = (
            f"Expected vocab={expected_vocab_size} but model has {cfg_vocab}. "
            "Fix via external surgery; runtime resizing is disabled."
        )
        if strict:
            raise ValueError(msg)
        logger.warning(msg)


def load_student_hf(
    path_or_id: str,
    *,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
) -> Tuple[nn.Module, object]:
    """
    Load a HuggingFace-style student folder or repo ID.
    Expects: config.json, model.safetensors(.index + shards) OR pytorch_model.bin/pt, tokenizer files.
    """
    # Tokenizer first so it can be the source of truth for padding.
    tok = AutoTokenizer.from_pretrained(path_or_id, use_fast=True, trust_remote_code=trust_remote_code)
    if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
        tok.pad_token = tok.eos_token  # Llama-style padding

    dtype = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16,   "fp16": torch.float16,
        "float32": torch.float32,   "fp32": torch.float32,
    }.get(torch_dtype.lower(), torch.bfloat16)

    model = AutoModelForCausalLM.from_pretrained(
        path_or_id,
        use_safetensors=True,
        low_cpu_mem_usage=True,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    _retie_if_possible(model)
    _sanity_readout(model, tok)
    return model, tok


# -------------------------
# Optional: .pt override inside HF folder (rarely needed)
# -------------------------
def maybe_override_with_pt_weights(
    model: nn.Module,
    hf_folder: str,
    *,
    force: bool = False,
) -> None:
    """
    If a pytorch_model.pt is present in the same HF folder and `force=True`,
    load that state_dict over the AutoModel weights. Not used by default.
    """
    if not force:
        return
    pt_path = Path(hf_folder) / "pytorch_model.pt"
    if not pt_path.exists():
        logger.info("No pytorch_model.pt in %s; skipping .pt override", hf_folder)
        return
    state = torch.load(str(pt_path), map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    logger.info("Loaded .pt override with %d missing and %d unexpected keys", len(missing), len(unexpected))


# -------------------------
# Public API
# -------------------------
def save_frozen_mask(
    run_dir: str,
    block_names: Iterable[str],
    *,
    output_gcs_uri: Optional[str] = None,
    run_id: Optional[str] = None,
) -> Path:
    ensure_dir(run_dir)
    path = Path(run_dir) / "frozen_mask.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump({"trainable_blocks": list(block_names)}, f, indent=2, sort_keys=True)

    if output_gcs_uri and run_id:
        from .gcs_io import local_to_gcs
        dest = f"{output_gcs_uri.rstrip('/')}/{run_id}/frozen_mask.json"
        local_to_gcs(str(path), dest)
    return path


def initialize_student(
    student_model_path: str,
    run_local_dir: str,
    seq_len: int,  # kept for signature parity; not used for HF load
    *,
    output_gcs_uri: Optional[str] = None,
    run_id: Optional[str] = None,
    expected_vocab_size: Optional[int] = None,  # e.g., 128_256 for Llama-3.x
    strict_vocab_check: bool = False,
    torch_dtype: str = "bfloat16",
    trust_remote_code: bool = False,
    force_pt_override: bool = False,
) -> Tuple[nn.Module, object]:
    """
    Initialize the student from a local HF folder (unzipped by the runner) or HF repo ID.
    - No runtime vocab resizing here; tokenizer is the source of truth.
    - Optionally enforce a strict vocab match with `expected_vocab_size`.
    - Optionally override with a local pytorch_model.pt located in the same folder.
    Returns (model, tokenizer).
    """
    # Enable fused kernels if available (does not change semantics).
    enable_flash_attn_if_available()

    model, tok = load_student_hf(
        student_model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    if force_pt_override:
        maybe_override_with_pt_weights(model, student_model_path, force=True)
        _retie_if_possible(model)  # re-tie after manual load

    _check_vocab(model, tok, expected_vocab_size=expected_vocab_size, strict=strict_vocab_check)
    save_frozen_mask(run_local_dir, _FROZEN_BLOCK_NAMES, output_gcs_uri=output_gcs_uri, run_id=run_id)
    return model, tok


# -------------------------
# Grad checkpointing wrapper
# -------------------------
def apply_grad_checkpointing(model: nn.Module, enabled: bool) -> nn.Module:
    if not enabled:
        return model
    import torch.utils.checkpoint as cp

    # Best-effort wrapping for common HF architectures that expose .model.layers
    wrapped = 0
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if getattr(layer, "_checkpoint_wrapped", False):
                continue

            orig_forward = layer.forward

            def _mk_wrapped(fwd):
                def _wrapped(*args, **kwargs):
                    def inner(*ip):
                        return fwd(*ip, **kwargs)
                    return cp.checkpoint(inner, *args, use_reentrant=False)
                return _wrapped

            layer.forward = _mk_wrapped(orig_forward)  # type: ignore[assignment]
            setattr(layer, "_checkpoint_wrapped", True)
            wrapped += 1

    # Fallback: common custom stacks expose .blocks
    elif hasattr(model, "blocks"):
        for block in model.blocks:
            if getattr(block, "_checkpoint_wrapped", False):
                continue

            orig_forward = block.forward

            def _mk_wrapped(fwd):
                def _wrapped(*args, **kwargs):
                    def inner(*ip):
                        return fwd(*ip, **kwargs)
                    return cp.checkpoint(inner, *args, use_reentrant=False)
                return _wrapped

            block.forward = _mk_wrapped(orig_forward)  # type: ignore[assignment]
            setattr(block, "_checkpoint_wrapped", True)
            wrapped += 1

    logger.info("Gradient checkpointing wrapped %d layers", wrapped)
    return model
