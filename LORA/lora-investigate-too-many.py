from huggingface_hub import HfApi, model_info, dataset_info
import re
import csv
from tqdm import tqdm  # optional, for progress bar

ORG = "Lots-of-LoRAs"
MODEL_PREFIX = "Mistral-7B-Instruct-v0.2-4b-r16-task"

api = HfApi()


def extract_task_id(model_id: str):
    """
    Example: 'Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task581'
    -> '581'
    """
    m = re.search(r"-task(\d+)$", model_id)
    return m.group(1) if m else None


def guess_dataset_id(task_id: str):
    # Datasets are named like: Lots-of-LoRAs/task581_socialiqa_question_generation
    # We'll search for "task{ID}_" in the dataset name.
    all_ds = api.list_datasets(author=ORG, search=f"task{task_id}_")
    for ds in all_ds:
        # strict match on the prefix "task{ID}_"
        name = ds.id.split("/")[-1]
        if name.startswith(f"task{task_id}_"):
            return ds.id
    return None


def main():
    print("Listing models...")
    models = api.list_models(author=ORG, limit=2000)

    rows = []
    for m in tqdm(models):
        mid = m.id  # e.g. Lots-of-LoRAs/Mistral-7B-Instruct-v0.2-4b-r16-task581
        if MODEL_PREFIX not in mid:
            # Skip anything that isn't one of the standard Mistral LoRAs
            continue

        task_id = extract_task_id(mid)
        if task_id is None:
            continue

        # Try to get more metadata from the model card
        try:
            mi = model_info(mid)
            base_model = mi.base_model if hasattr(mi, "base_model") else None
        except Exception:
            mi = None
            base_model = None

        # Match dataset
        ds_id = guess_dataset_id(task_id)
        ds_name = ds_task_type = ds_size = None

        if ds_id:
            try:
                di = dataset_info(ds_id)
                ds_name = ds_id.split("/")[-1]
                # common bits of metadata are stored in "card_data"
                card = (di.card_data or {})
                ds_task_type = card.get("task", card.get("tasks"))
                ds_size = card.get("size", card.get("num_examples"))
            except Exception:
                pass

        rows.append({
            "model_id": mid,
            "task_id": task_id,
            "base_model": base_model,
            "dataset_id": ds_id,
            "dataset_name": ds_name,
            "dataset_task_type": ds_task_type,
            "dataset_size": ds_size,
        })

    out_file = "lots_of_loras_index.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {out_file}")


if __name__ == "__main__":
    main()
