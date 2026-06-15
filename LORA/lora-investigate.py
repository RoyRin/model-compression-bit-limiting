from huggingface_hub import HfApi, model_info
import re
import csv
import time
from tqdm import tqdm

ORG = "Lots-of-LoRAs"
MODEL_PREFIX = "Mistral-7B-Instruct-v0.2-4b-r16-task"

api = HfApi()


def extract_task_id(model_id: str):
    m = re.search(r"-task(\d+)$", model_id)
    return m.group(1) if m else None


def build_dataset_index():
    """
    Build a mapping: task_id -> dataset_id
    using ONE API call.
    """
    print("Listing datasets (once)...")
    datasets = api.list_datasets(author=ORG, limit=2000)

    index = {}
    for ds in datasets:
        # ds.id = Lots-of-LoRAs/task581_socialiqa_question_generation
        name = ds.id.split("/")[-1]
        m = re.match(r"task(\d+)_", name)
        if m:
            task_id = m.group(1)
            index[task_id] = ds.id

    print(f"Indexed {len(index)} datasets")
    return index


def main():
    dataset_index = build_dataset_index()

    print("Listing models...")
    models = api.list_models(author=ORG, limit=2000)

    rows = []
    for m in tqdm(models):
        mid = m.id
        if MODEL_PREFIX not in mid:
            continue

        task_id = extract_task_id(mid)
        if task_id is None:
            continue

        # Model metadata (optional; this is also an API call)
        try:
            mi = model_info(mid)
            base_model = getattr(mi, "base_model", None)
            time.sleep(0.05)  # VERY light throttle
        except Exception:
            base_model = None

        ds_id = dataset_index.get(task_id)

        rows.append({
            "model_id": mid,
            "task_id": task_id,
            "base_model": base_model,
            "dataset_id": ds_id,
        })

    with open("lots_of_loras_index.csv", "w", newline="",
              encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model_id", "task_id", "base_model", "dataset_id"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows")


if __name__ == "__main__":
    main()
