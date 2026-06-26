import argparse
import json

from cmafnet.cohort_manifest import (
    assign_cohort,
    build_records,
    patient_level_split,
    save_split_manifest,
)
from cmafnet.runtime_paths import external_manifest_path, project_root, split_manifest_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=project_root())
    args = parser.parse_args()
    records = build_records(args.root)
    internal, external = assign_cohort(records, external_manifest_path())
    buckets = patient_level_split(internal)
    save_split_manifest(split_manifest_path(), buckets)
    payload = {
        "internal_records": len(internal),
        "external_records": len(external),
        "train": len(buckets["train"]),
        "val": len(buckets["val"]),
        "test": len(buckets["test"]),
    }
    with open(split_manifest_path().replace(".json", "_summary.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    main()
