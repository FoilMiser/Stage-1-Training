"""Command line driver for dataset preprocessing pipelines."""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
from typing import Dict, Iterable, List, Tuple

import yaml

from . import io
from .pipelines import (
    asdiv,
    c4,
    codesearchnet,
    dolma,
    fineweb_edu,
    gsm8k,
    svamp,
    wikipedia,
    wikitext,
)

LOGGER = logging.getLogger(__name__)

CONFIG_MAP = {
    "wikitext": ["wikitext"],
    "wikipedia": ["wikipedia"],
    "c4": ["c4_train", "c4_validation"],
    "fineweb_edu": ["fineweb_edu_10bt", "fineweb_edu_100bt", "fineweb_edu_350bt"],
    "dolma": ["dolma_v1_6_sample"],
    "gsm8k": ["gsm8k"],
    "svamp": ["svamp"],
    "asdiv": ["asdiv"],
    "codesearchnet": ["codesearchnet"],
}

PIPELINE_MODULES = {
    "wikitext": wikitext,
    "wikipedia": wikipedia,
    "c4_train": c4,
    "c4_validation": c4,
    "fineweb_edu_10bt": fineweb_edu,
    "fineweb_edu_100bt": fineweb_edu,
    "fineweb_edu_350bt": fineweb_edu,
    "dolma_v1_6_sample": dolma,
    "gsm8k": gsm8k,
    "svamp": svamp,
    "asdiv": asdiv,
    "codesearchnet": codesearchnet,
}


def load_datasets_config() -> Dict[str, dict]:
    config_path = pathlib.Path(__file__).resolve().parent / "config" / "datasets.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _ensure_list(obj) -> List[str]:
    if obj is None:
        return []
    if isinstance(obj, (list, tuple)):
        return list(obj)
    return [obj]


def _prepare_inputs(inputs: Iterable[str], tmp_dir: str) -> List[Tuple[str, str]]:
    local_inputs: List[Tuple[str, str]] = []
    os.makedirs(tmp_dir, exist_ok=True)
    for idx, uri in enumerate(inputs):
        if uri.startswith("gs://"):
            filename = os.path.basename(uri.rstrip("/")) or f"input-{idx}"
            local_path = os.path.join(tmp_dir, filename)
            LOGGER.info("Downloading %s to %s", uri, local_path)
            io.gcs_to_local(uri, local_path)
        else:
            local_path = uri
        local_inputs.append((uri, local_path))
    return local_inputs


def _upload_outputs(local_shard_dir: str, gcs_out: str) -> List[str]:
    uploads: List[str] = []
    for path in sorted(pathlib.Path(local_shard_dir).glob("*.jsonl")):
        uploads.append(io.local_to_gcs(str(path), gcs_out))
    return uploads


def _resolve_manifest(entry: dict, override_type: str | None, override_weight: float | None) -> str | None:
    manifest = entry.get("manifest")
    if manifest is None:
        return None
    data = json.loads(manifest)
    if override_type:
        data["type"] = override_type
    if override_weight is not None:
        data["weight"] = override_weight
    return json.dumps(data, separators=(",", ":"))


def run_dataset(
    dataset_key: str,
    entry: dict,
    *,
    override_type: str | None,
    override_weight: float | None,
    in_override: str | None,
    out_override: str | None,
    max_records: int,
    tmp_root: str,
) -> dict:
    module = PIPELINE_MODULES[dataset_key]
    dataset_type = override_type or entry.get("type", "lm")
    inputs = _ensure_list(in_override or entry.get("in"))
    output_uri = out_override or entry.get("out")
    if not output_uri:
        raise ValueError(f"No output destination for dataset {dataset_key}")

    dataset_tmp = os.path.join(tmp_root, dataset_key)
    work_tmp = os.path.join(dataset_tmp, "work")
    local_inputs = _prepare_inputs(inputs, os.path.join(dataset_tmp, "inputs"))
    local_output_dir = os.path.join(dataset_tmp, "shards")
    os.makedirs(work_tmp, exist_ok=True)

    LOGGER.info("Running pipeline %s", dataset_key)
    summary = module.run(
        local_inputs,
        local_output_dir,
        dataset_type=dataset_type,
        max_records=max_records,
        work_dir=work_tmp,
    )

    uploads = _upload_outputs(local_output_dir, output_uri)
    manifest_line = _resolve_manifest(entry, override_type, override_weight)

    first_shard = os.path.basename(summary["shards"][0]) if summary["shards"] else "n/a"
    last_shard = os.path.basename(summary["shards"][-1]) if summary["shards"] else "n/a"
    LOGGER.info(
        "Dataset %s processed %s records into %s shards (%s -> %s)",
        dataset_key,
        summary["records"],
        len(summary["shards"]),
        first_shard,
        last_shard,
    )

    return {
        "dataset": dataset_key,
        "records": summary["records"],
        "shards": summary["shards"],
        "uploads": uploads,
        "manifest": manifest_line,
        "first_shard": first_shard,
        "last_shard": last_shard,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dataset preprocessing toolkit")
    parser.add_argument(
        "--job",
        required=True,
        choices=[
            "wikitext",
            "wikipedia",
            "c4",
            "fineweb_edu",
            "dolma",
            "gsm8k",
            "svamp",
            "asdiv",
            "codesearchnet",
            "all",
        ],
    )
    parser.add_argument("--in", dest="input_path", help="Override input GCS URI", default=None)
    parser.add_argument("--out", dest="output_path", help="Override output GCS directory", default=None)
    parser.add_argument("--type", dest="sample_type", choices=["lm", "math_tool", "code"], default=None)
    parser.add_argument("--max-records", dest="max_records", type=int, default=20000)
    parser.add_argument("--manifest", action="store_true", help="Print manifest line(s)")
    parser.add_argument("--weight", dest="weight", type=float, default=None)
    parser.add_argument("--tmp", dest="tmp_dir", default="/tmp/prep")
    parser.add_argument("--log-level", dest="log_level", default="INFO")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    config = load_datasets_config()

    if args.job == "all":
        dataset_keys = list(config.keys())
    else:
        dataset_keys = CONFIG_MAP[args.job]

    results = []
    for key in dataset_keys:
        entry = config[key]
        in_override = args.input_path
        out_override = args.output_path
        if args.input_path and len(dataset_keys) > 1:
            raise ValueError("--in override is ambiguous for multi-dataset jobs")
        if args.output_path and len(dataset_keys) > 1:
            raise ValueError("--out override is ambiguous for multi-dataset jobs")
        result = run_dataset(
            key,
            entry,
            override_type=args.sample_type,
            override_weight=args.weight,
            in_override=in_override,
            out_override=out_override,
            max_records=args.max_records,
            tmp_root=args.tmp_dir,
        )
        results.append(result)
        if args.manifest and result["manifest"]:
            print(result["manifest"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
