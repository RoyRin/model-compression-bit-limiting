#!/usr/bin/env python3
"""
Download LoRAs and datasets from Hugging Face.

Supports parallel downloads for faster throughput.
"""

import yaml
import argparse
import time
from pathlib import Path
from huggingface_hub import snapshot_download
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor, as_completed
import os


def download_single_task(task_id, info, lora_dir, dataset_dir, skip_existing):
    """Download a single LoRA and dataset pair.

    Returns:
        (task_id, success, elapsed_time, error_msg)
    """
    start_time = time.time()

    lora_path = info['lora_path']
    dataset_path = info['dataset_path']

    # Extract repo IDs from URLs
    lora_repo = lora_path.replace('https://huggingface.co/', '')
    dataset_repo = dataset_path.replace('https://huggingface.co/datasets/', '')

    # Local paths
    local_lora_path = lora_dir / f"task{task_id}"
    local_dataset_path = dataset_dir / f"task{task_id}"

    # Check if already exists
    if skip_existing and local_lora_path.exists(
    ) and local_dataset_path.exists():
        return (task_id, True, 0, "skipped (exists)")

    try:
        # Download LoRA
        if not local_lora_path.exists():
            snapshot_download(
                repo_id=lora_repo,
                local_dir=str(local_lora_path),
                local_dir_use_symlinks=False,
            )

        # Download dataset
        if not local_dataset_path.exists():
            dataset = load_dataset(dataset_repo)
            dataset.save_to_disk(str(local_dataset_path))

        elapsed = time.time() - start_time
        return (task_id, True, elapsed, None)

    except Exception as e:
        elapsed = time.time() - start_time
        return (task_id, False, elapsed, str(e))


def download_loras_and_datasets(
    yaml_path: str,
    output_dir: str,
    num_to_download: int = 100,
    skip_existing: bool = True,
    num_workers: int = 4,
):
    """Download LoRAs and datasets from the YAML index.

    Args:
        yaml_path: Path to the YAML index file
        output_dir: Directory to save downloads
        num_to_download: Number of LoRA/dataset pairs to download
        skip_existing: Skip if already downloaded
        num_workers: Number of parallel download workers
    """
    total_start_time = time.time()

    output_dir = Path(output_dir)
    lora_dir = output_dir / "loras"
    dataset_dir = output_dir / "datasets"

    lora_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Load YAML index
    with open(yaml_path, 'r') as f:
        tasks = yaml.safe_load(f)

    # Select tasks to download
    tasks_to_download = list(tasks.items())[:num_to_download]

    print(f"Loaded {len(tasks)} tasks from {yaml_path}")
    print(f"Will download {len(tasks_to_download)} LoRA/dataset pairs")
    print(f"Output directory: {output_dir}")
    print(f"Parallel workers: {num_workers}")
    print("=" * 60)

    downloaded = 0
    skipped = 0
    failed = []
    download_times = []

    def format_time(seconds):
        """Format seconds as mm:ss or hh:mm:ss."""
        if seconds < 3600:
            return f"{int(seconds//60)}:{int(seconds%60):02d}"
        else:
            return f"{int(seconds//3600)}:{int((seconds%3600)//60):02d}:{int(seconds%60):02d}"

    if num_workers > 1:
        # Parallel download
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(download_single_task, task_id, info, lora_dir, dataset_dir, skip_existing):
                (task_id, info)
                for task_id, info in tasks_to_download
            }

            for i, future in enumerate(as_completed(futures)):
                task_id, success, elapsed, error = future.result()
                info = tasks[task_id]
                cumulative = time.time() - total_start_time

                if error == "skipped (exists)":
                    skipped += 1
                    print(
                        f"[{i+1}/{len(tasks_to_download)}] [{format_time(cumulative)}] Task {task_id}: skipped (exists)"
                    )
                elif success:
                    downloaded += 1
                    download_times.append(elapsed)
                    print(
                        f"[{i+1}/{len(tasks_to_download)}] [{format_time(cumulative)}] Task {task_id}: ✓ ({elapsed:.1f}s) - {info['description'][:40]}"
                    )
                else:
                    failed.append((task_id, error))
                    print(
                        f"[{i+1}/{len(tasks_to_download)}] [{format_time(cumulative)}] Task {task_id}: ✗ ({elapsed:.1f}s) - {error[:50]}"
                    )
    else:
        # Sequential download
        for i, (task_id, info) in enumerate(tasks_to_download):
            task_id, success, elapsed, error = download_single_task(
                task_id, info, lora_dir, dataset_dir, skip_existing)
            cumulative = time.time() - total_start_time

            if error == "skipped (exists)":
                skipped += 1
                print(
                    f"[{i+1}/{len(tasks_to_download)}] [{format_time(cumulative)}] Task {task_id}: skipped (exists)"
                )
            elif success:
                downloaded += 1
                download_times.append(elapsed)
                print(
                    f"[{i+1}/{len(tasks_to_download)}] [{format_time(cumulative)}] Task {task_id}: ✓ ({elapsed:.1f}s) - {info['description'][:40]}"
                )
            else:
                failed.append((task_id, error))
                print(
                    f"[{i+1}/{len(tasks_to_download)}] [{format_time(cumulative)}] Task {task_id}: ✗ ({elapsed:.1f}s) - {error[:50]}"
                )

    total_elapsed = time.time() - total_start_time

    print("\n" + "=" * 60)
    print(f"Download complete!")
    print(f"  Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Successfully downloaded: {downloaded}")
    print(f"  Skipped (already exists): {skipped}")
    print(f"  Failed: {len(failed)}")

    if download_times:
        avg_time = sum(download_times) / len(download_times)
        print(f"  Avg time per download: {avg_time:.1f}s")

    if failed:
        print(f"\nFailed downloads:")
        for task_id, error in failed:
            print(f"  Task {task_id}: {error}")

    # Save manifest of downloaded items
    manifest_path = output_dir / "manifest.yaml"
    manifest = {
        'num_downloaded': downloaded,
        'lora_dir': str(lora_dir),
        'dataset_dir': str(dataset_dir),
        'tasks': {}
    }

    for task_id, info in list(tasks.items())[:downloaded]:
        local_lora = lora_dir / f"task{task_id}"
        local_dataset = dataset_dir / f"task{task_id}"
        if local_lora.exists() and local_dataset.exists():
            manifest['tasks'][task_id] = {
                'lora_path': str(local_lora),
                'dataset_path': str(local_dataset),
                'description': info['description'],
                'hf_lora': info['lora_path'],
                'hf_dataset': info['dataset_path'],
            }

    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f, default_flow_style=False)
    print(f"\nManifest saved to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Download LoRAs and datasets")
    parser.add_argument("--yaml",
                        type=str,
                        default="LORA/lots_of_loras_index.yaml",
                        help="Path to YAML index file")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/n/netscratch/sham_lab/Lab/rrinberg/compression/LORAS",
        help="Directory to save downloads")
    parser.add_argument("--num",
                        type=int,
                        default=100,
                        help="Number of LoRA/dataset pairs to download")
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)")
    parser.add_argument("--no-skip-existing",
                        action="store_true",
                        help="Re-download even if already exists")

    args = parser.parse_args()

    download_loras_and_datasets(
        yaml_path=args.yaml,
        output_dir=args.output_dir,
        num_to_download=args.num,
        skip_existing=not args.no_skip_existing,
        num_workers=args.workers,
    )


if __name__ == "__main__":
    main()
