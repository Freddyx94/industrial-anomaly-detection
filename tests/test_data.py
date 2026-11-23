from pathlib import Path

import pytest

from inspection.data import check_structure, index_dataset, list_categories


def test_list_categories_ignores_hidden_files(tiny_mvtec: Path) -> None:
    assert list_categories(tiny_mvtec) == ["widget"]


def test_missing_dataset_gives_clear_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset folder not found"):
        list_categories(tmp_path / "nope")


def test_index_counts_and_labels(tiny_mvtec: Path) -> None:
    index = index_dataset(tiny_mvtec)
    assert len(index) == 5
    counts = index.groupby(["split", "defect_type"]).size().to_dict()
    assert counts == {("test", "good"): 1, ("test", "scratch"): 2, ("train", "good"): 2}
    assert (index[index["defect_type"] == "good"]["label"] == 0).all()
    assert (index[index["defect_type"] == "scratch"]["label"] == 1).all()


def test_masks_only_for_defects(tiny_mvtec: Path) -> None:
    index = index_dataset(tiny_mvtec)
    assert index[index["label"] == 0]["mask_path"].isna().all()
    defect_masks = index[index["label"] == 1]["mask_path"]
    assert all(p.name.endswith("_mask.png") and p.exists() for p in defect_masks)


def test_check_structure_passes_on_valid_tree(tiny_mvtec: Path) -> None:
    assert check_structure(index_dataset(tiny_mvtec), tiny_mvtec) == []


def test_check_structure_finds_missing_mask(tiny_mvtec: Path) -> None:
    (tiny_mvtec / "widget/ground_truth/scratch/001_mask.png").unlink()
    problems = check_structure(index_dataset(tiny_mvtec), tiny_mvtec)
    assert problems == ["widget: 1 defective images have no mask"]


def test_check_structure_finds_defect_in_train(tiny_mvtec: Path) -> None:
    bad = tiny_mvtec / "widget/train/scratch/000.png"
    bad.parent.mkdir()
    bad.write_bytes((tiny_mvtec / "widget/train/good/000.png").read_bytes())
    problems = check_structure(index_dataset(tiny_mvtec), tiny_mvtec)
    assert "widget: training set contains non-good images" in problems
