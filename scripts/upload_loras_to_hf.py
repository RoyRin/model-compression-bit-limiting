#!/usr/bin/env python3
"""Upload all LoRA adapters and cluster data to HuggingFace dataset repo.

Uploads to: https://huggingface.co/datasets/royrin/model-compression

Uses upload_folder per directory (one commit per directory, ~10 total).
Retries on 429 rate limits with exponential backoff.
"""

import argparse
import time
from pathlib import Path
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

REPO_ID = "royrin/model-compression"
BASE = Path("/n/netscratch/sham_lab/Lab/rrinberg/compression")

UPLOAD_DIRS = {
    # LoRAs
    "loras/lmsys": BASE / "lmsys-loras",
    "loras/wildchat": BASE / "wildchat-loras",
    "loras/enwik9-10": BASE / "enwik9-loras",
    "loras/enwik9-50": BASE / "enwik9-loras-50",
    "loras/enwik9-llama-base-10": BASE / "enwik9-loras-llama-base",
    "loras/enwik9-llama-instruct-10": BASE / "enwik9-loras-llama-instruct",
    # Qwen3-4B LoRAs
    "loras/lmsys-qwen3-4b": BASE / "lmsys-loras-qwen3-4b",
    "loras/wildchat-qwen3-4b": BASE / "wildchat-loras-qwen3-4b",
    "loras/enwik9-qwen3-4b": BASE / "enwik9-loras-qwen3-4b",
    "loras/enwik9-50-qwen3-4b": BASE / "enwik9-loras-50-qwen3-4b",
    # Cluster data
    "data/lmsys": BASE / "lmsys-clustered" / "clusters",
    "data/wildchat": BASE / "wildchat-clustered" / "clusters",
    "data/enwik9-10": BASE / "enwik9-clustered" / "clusters",
    "data/enwik9-50": BASE / "enwik9-clustered-50" / "clusters",
}

MAX_RETRIES = 5
INITIAL_BACKOFF = 120  # seconds


def upload_dir(api, local_dir, hf_prefix):
    """Upload an entire directory with retry on 429."""
    for attempt in range(MAX_RETRIES):
        try:
            api.upload_folder(
                folder_path=str(local_dir),
                path_in_repo=hf_prefix,
                repo_id=REPO_ID,
                repo_type="dataset",
                ignore_patterns=["**/checkpoint-*/**", "**/checkpoint-*"],
                commit_message=f"Upload {hf_prefix}",
            )
            return
        except HfHubHTTPError as e:
            if "429" in str(e) and attempt < MAX_RETRIES - 1:
                wait = INITIAL_BACKOFF * (2**attempt)
                print(
                    f"  Rate limited, waiting {wait}s before retry ({attempt+1}/{MAX_RETRIES})...",
                    flush=True)
                time.sleep(wait)
            else:
                raise


def count_files(local_dir):
    """Count non-checkpoint files and their total size."""
    files = [
        f for f in local_dir.rglob("*") if f.is_file() and not any(
            p.startswith("checkpoint-")
            for p in f.relative_to(local_dir).parts)
    ]
    total_size = sum(f.stat().st_size for f in files)
    return len(files), total_size


def main():
    parser = argparse.ArgumentParser(
        description="Upload LoRAs and data to HuggingFace")
    parser.add_argument("--dry-run",
                        action="store_true",
                        help="List what would be uploaded")
    parser.add_argument("--only",
                        type=str,
                        nargs="+",
                        help="Only upload these keys (e.g. 'loras/lmsys')")
    parser.add_argument("--loras-only",
                        action="store_true",
                        help="Only upload LoRAs")
    parser.add_argument("--data-only",
                        action="store_true",
                        help="Only upload cluster data")
    parser.add_argument("--skip", type=str, nargs="+", help="Skip these keys")
    args = parser.parse_args()

    api = HfApi()

    dirs = dict(UPLOAD_DIRS)
    if args.loras_only:
        dirs = {k: v for k, v in dirs.items() if k.startswith("loras/")}
    elif args.data_only:
        dirs = {k: v for k, v in dirs.items() if k.startswith("data/")}
    if args.only:
        unknown = [k for k in args.only if k not in UPLOAD_DIRS]
        if unknown:
            print(f"Unknown key(s): {unknown}")
            print(f"Available: {list(UPLOAD_DIRS.keys())}")
            return
        dirs = {k: UPLOAD_DIRS[k] for k in args.only}
    if args.skip:
        dirs = {k: v for k, v in dirs.items() if k not in args.skip}

    # Summary
    grand_files = 0
    grand_size = 0
    for hf_prefix, local_dir in dirs.items():
        if local_dir.exists():
            nf, sz = count_files(local_dir)
            grand_files += nf
            grand_size += sz
    print(
        f"Total: {grand_files} files, {grand_size / 1e9:.1f} GB across {len(dirs)} directories"
    )
    print(f"Will make {len(dirs)} commits\n", flush=True)

    if args.dry_run:
        for hf_prefix, local_dir in dirs.items():
            if not local_dir.exists():
                print(f"  SKIP {hf_prefix}: does not exist")
                continue
            nf, sz = count_files(local_dir)
            print(f"  {hf_prefix}: {nf} files, {sz/1e9:.1f} GB")
        return

    uploaded_size = 0
    overall_start = time.time()

    for i, (hf_prefix, local_dir) in enumerate(dirs.items()):
        if not local_dir.exists():
            print(f"[{i+1}/{len(dirs)}] SKIP {hf_prefix}: does not exist",
                  flush=True)
            continue

        nf, sz = count_files(local_dir)
        print(
            f"[{i+1}/{len(dirs)}] Uploading {hf_prefix}: {nf} files, {sz/1e9:.1f} GB ...",
            flush=True)

        t0 = time.time()
        upload_dir(api, local_dir, hf_prefix)
        elapsed = time.time() - t0

        uploaded_size += sz
        elapsed_total = time.time() - overall_start
        remaining = grand_size - uploaded_size
        speed = uploaded_size / elapsed_total / 1e6 if elapsed_total > 0 else 0
        eta_min = (remaining / (speed * 1e6) / 60) if speed > 0 else 0

        print(
            f"  Done in {elapsed/60:.1f} min | "
            f"Overall: {uploaded_size/1e9:.1f}/{grand_size/1e9:.1f} GB | "
            f"{speed:.0f} MB/s | ETA: {eta_min:.0f} min\n",
            flush=True)

    total_elapsed = time.time() - overall_start
    print(
        f"All uploads complete! {uploaded_size/1e9:.1f} GB in {total_elapsed/60:.1f} min"
    )


if __name__ == "__main__":
    main()
