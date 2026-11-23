"""Index the MVTec AD folder into one table.

Why: every stage needs the same answer to "which images, which split, which label, which mask".
Building it once as a DataFrame (one row per image) means the notebook, training code and
evaluation code all use the same list, and we can count and filter it like a SQL table.

Expected layout (official MVTec AD):
    <root>/<category>/train/good/*.png                      defect-free training images
    <root>/<category>/test/good/*.png                       defect-free test images
    <root>/<category>/test/<defect_type>/*.png              defective test images
    <root>/<category>/ground_truth/<defect_type>/*_mask.png pixel masks (white = defect)
"""

from pathlib import Path

import pandas as pd

GOOD = "good"
SPLITS = ("train", "test")


def list_categories(root: Path) -> list[str]:
    """Category folder names under root, sorted (ignores hidden files like .DS_Store)."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}. Put MVTec AD there.")
    return sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))


def mask_path_for(root: Path, category: str, defect_type: str, image_path: Path) -> Path:
    """Where MVTec stores the mask of a defective test image: <stem>_mask.png."""
    return Path(root) / category / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"


def index_dataset(root: Path, categories: list[str] | None = None) -> pd.DataFrame:
    """Return one row per image.

    Columns: category, split ("train"/"test"), defect_type ("good" or e.g. "scratch"),
    label (0 = good, 1 = defective), image_path, mask_path (None for good images, because
    MVTec only stores masks for defects; a good image's mask would be all black).
    """
    root = Path(root)
    categories = categories or list_categories(root)
    rows = []
    for category in categories:
        for split in SPLITS:
            split_dir = root / category / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Missing folder: {split_dir}")
            for type_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
                is_good = type_dir.name == GOOD
                for image_path in sorted(type_dir.glob("*.png")):
                    rows.append(
                        {
                            "category": category,
                            "split": split,
                            "defect_type": type_dir.name,
                            "label": 0 if is_good else 1,
                            "image_path": image_path,
                            "mask_path": None
                            if is_good
                            else mask_path_for(root, category, type_dir.name, image_path),
                        }
                    )
    return pd.DataFrame(rows)


def check_structure(index: pd.DataFrame, root: Path) -> list[str]:
    """List problems with the dataset; an empty list means the structure is as expected.

    Checks per category: training images exist and are all good; the test set has both good and
    defective images; every defective image has a mask; no mask is left without an image.
    """
    problems: list[str] = []
    for category, rows in index.groupby("category"):
        train = rows[rows["split"] == "train"]
        test = rows[rows["split"] == "test"]
        if train.empty:
            problems.append(f"{category}: no training images")
        if (train["label"] != 0).any():
            problems.append(f"{category}: training set contains non-good images")
        if (test["label"] == 0).sum() == 0:
            problems.append(f"{category}: no good test images")
        if (test["label"] == 1).sum() == 0:
            problems.append(f"{category}: no defective test images")

        expected_masks = set(rows["mask_path"].dropna())
        missing = [p for p in expected_masks if not p.exists()]
        if missing:
            problems.append(f"{category}: {len(missing)} defective images have no mask")
        on_disk = set((Path(root) / category / "ground_truth").rglob("*.png"))
        orphans = on_disk - expected_masks
        if orphans:
            problems.append(f"{category}: {len(orphans)} masks have no matching image")
    return problems
