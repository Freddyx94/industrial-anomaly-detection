"""Tests for config loading and the device/seed helpers."""

import torch

from inspection.config import PROJECT_ROOT, load_config
from inspection.device import get_device, set_seed


def test_config_categories_are_real_mvtec_categories() -> None:
    config = load_config()
    all_categories = set(config["dataset"]["objects"]) | set(config["dataset"]["textures"])
    assert len(all_categories) == 15
    assert set(config["categories"]) <= all_categories


def test_project_root_holds_config() -> None:
    assert (PROJECT_ROOT / "config.yaml").is_file()


def test_get_device_returns_a_known_device() -> None:
    assert get_device().type in {"cuda", "mps", "cpu"}


def test_set_seed_makes_random_numbers_repeat() -> None:
    set_seed(7)
    first = torch.rand(3)
    set_seed(7)
    assert torch.equal(first, torch.rand(3))
