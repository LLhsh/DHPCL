"""Create a patient-disjoint 30/5/10 split without changing any HDF5 data."""

import json
import re
from pathlib import Path


def read_list(path):
    entries = path.read_text(encoding="utf-8").splitlines()
    if not entries or any(not entry for entry in entries):
        raise ValueError(f"Empty list or blank entry: {path}")
    if len(entries) != len(set(entries)):
        raise ValueError(f"Duplicate entries: {path}")
    return entries


def slice_case(name):
    match = re.fullmatch(r"(patient\d+_LGE)_slice_\d+", name)
    if match is None:
        raise ValueError(f"Unexpected slice name: {name}")
    return match.group(1)


def main():
    root = Path(__file__).resolve().parents[1] / "datasets" / "mscmrseg"
    backup = root / "split_backup_before_30_5_10"
    filenames = ("train_slices.list", "val.list", "test.list")
    source = backup if backup.exists() else root
    originals = {name: (source / name).read_bytes() for name in filenames}
    original_slices = read_list(source / "train_slices.list")
    original_train = list(dict.fromkeys(map(slice_case, original_slices)))
    original_val = read_list(source / "val.list")
    if len(original_train) != 35 or len(original_val) != 10:
        raise ValueError("Expected the original 35-training / 10-validation split")

    # Preserve the original patient order, including the first labeled patients.
    cases = {
        "train": original_train[:30],
        "val": original_train[30:],
        "test": original_val,
    }
    groups = {name: set(values) for name, values in cases.items()}
    if any(groups[a] & groups[b] for a, b in
           (("train", "val"), ("train", "test"), ("val", "test"))):
        raise ValueError("Patient overlap between splits")
    volumes = {path.stem for path in (root / "data").glob("*.h5")}
    expected = {f"patient{i}_LGE" for i in range(1, 46)}
    if volumes != expected or set().union(*groups.values()) != volumes:
        raise ValueError("The split must cover exactly the 45 available patients")

    available_slices = {
        path.stem for path in (root / "data" / "slices").glob("*.h5")
    }
    available_cases = {slice_case(name) for name in available_slices}
    if available_cases != volumes:
        raise ValueError("Slice files do not cover exactly the available patients")
    expected_original_slices = {
        name for name in available_slices if slice_case(name) in set(original_train)
    }
    if set(original_slices) != expected_original_slices:
        raise ValueError("Original training list does not match all training slice files")
    train_slices = [
        name for name in original_slices if slice_case(name) in groups["train"]
    ]
    outputs = {
        "train_slices.list": train_slices,
        "val.list": cases["val"],
        "test.list": cases["test"],
    }

    # Back up the original list bytes once; repeated runs use this same source.
    if not backup.exists():
        backup.mkdir()
        for name, content in originals.items():
            (backup / name).write_bytes(content)
    for name, entries in outputs.items():
        (root / name).write_bytes(("\n".join(entries) + "\n").encode("utf-8"))
    info = {
        "unit": "patient",
        "sequence": "LGE",
        "method": "First 30 original training patients for training; last 5 for "
                  "validation; original 10 validation patients for testing.",
        "original_lists_backup": backup.name,
        "cases": cases,
        "case_counts": {name: len(values) for name, values in cases.items()},
        "train_slice_count": len(train_slices),
        "available_slice_counts": {
            group: sum(slice_case(name) in members for name in available_slices)
            for group, members in groups.items()
        },
    }
    (root / "split_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )

    for name, entries in outputs.items():
        if read_list(root / name) != entries:
            raise ValueError(f"Written list verification failed: {name}")
    print(json.dumps(info, indent=2))
    print("SPLIT_OK: 30/5/10 patients, disjoint, all 45 covered, all paths exist")


if __name__ == "__main__":
    main()
