"""Tests for the YAML configuration loader."""

import pytest
import yaml

from downscale.config import Config, load_config


@pytest.fixture
def config_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({
        "region": {"name": "france"},
        "paths": {"pairs": "data/custom"},
        "target": {"years": [2015, 2017], "train_years": [2015, 2016]},
        "training": {"loss": "mse", "epochs": 3},
    }))
    return path


def test_defaults_apply_to_absent_sections(config_file):
    cfg = load_config(config_file)
    assert cfg.paths.pairs == "data/custom"
    assert cfg.paths.runs == "runs"
    assert cfg.evaluation.n_bootstrap == 200


def test_region_resolves_from_a_preset_or_an_explicit_box(config_file):
    assert load_config(config_file).region_box == (40.0, 53.0, -7.0, 11.0)
    cfg = load_config(config_file, region={"name": None, "box": [40, 50, 0, 10]})
    assert cfg.region_box == (40.0, 50.0, 0.0, 10.0)


def test_validation_years_are_those_after_the_training_years(config_file):
    assert load_config(config_file).val_years == [2017]


def test_overrides_replace_file_values(config_file):
    assert load_config(config_file, training={"epochs": 9}).training.epochs == 9
    assert load_config(config_file).training.loss == "mse"


def test_unknown_keys_and_sections_raise(config_file):
    with pytest.raises(ValueError, match="unknown key"):
        load_config(config_file, training={"nonexistent": 1})
    with pytest.raises(ValueError, match="unknown configuration section"):
        load_config(config_file, nonexistent={"a": 1})


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "absent.yaml")


def test_unknown_region_name_raises(config_file):
    with pytest.raises(ValueError, match="not one of"):
        load_config(config_file, region={"name": "atlantis", "box": None}).region_box


def test_shipped_config_parses():
    cfg = load_config("config.yaml")
    assert isinstance(cfg, Config)
    assert cfg.training.loss in ("crps", "mse")
    assert len(cfg.target.times) == 4