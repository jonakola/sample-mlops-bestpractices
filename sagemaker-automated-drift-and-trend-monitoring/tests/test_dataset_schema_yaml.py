"""Unit tests for the checked-in `src/config/dataset_schema.yaml`.

These tests parse the real, checked-in YAML file directly (not via
`src.config.schema`'s cached accessors) and assert its structure matches
Requirement 1's acceptance criteria: a top-level `dataset` key with the
required scalar columns and a non-empty `features` list, well-formed
`auxiliary_columns` entries, and a well-formed `split` section.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
"""

from pathlib import Path

import yaml

_SCHEMA_YAML_PATH = (
    Path(__file__).resolve().parent.parent / "src" / "config" / "dataset_schema.yaml"
)


def _load_raw_dataset_dict():
    """Parse the checked-in YAML file independently of schema.py."""
    with open(_SCHEMA_YAML_PATH, "r") as f:
        doc = yaml.safe_load(f)
    return doc["dataset"]


def test_schema_file_exists_at_expected_path():
    """Requirement 1.5: Dataset_Schema_File resides at
    src/config/dataset_schema.yaml."""
    assert _SCHEMA_YAML_PATH.exists()
    assert _SCHEMA_YAML_PATH.is_file()


def test_top_level_dataset_key_present():
    """Requirement 1.1: the parsed YAML has a top-level `dataset` key."""
    with open(_SCHEMA_YAML_PATH, "r") as f:
        doc = yaml.safe_load(f)

    assert isinstance(doc, dict)
    assert "dataset" in doc
    assert isinstance(doc["dataset"], dict)


def test_dataset_declares_required_scalar_columns():
    """Requirement 1.1: `dataset` declares identifier_column,
    timestamp_column, and target_column."""
    dataset = _load_raw_dataset_dict()

    for key in ("identifier_column", "timestamp_column", "target_column"):
        assert key in dataset, f"dataset is missing required key '{key}'"
        value = dataset[key]
        assert isinstance(value, str)
        assert value != ""


def test_features_list_is_non_empty():
    """Requirement 1.1: `dataset` declares an ordered, non-empty
    `features` list."""
    dataset = _load_raw_dataset_dict()

    assert "features" in dataset
    features = dataset["features"]
    assert isinstance(features, list)
    assert len(features) > 0


def test_every_feature_entry_has_name_and_type():
    """Requirement 1.2: every entry in `features` declares a `name` and
    a `type`."""
    dataset = _load_raw_dataset_dict()
    features = dataset["features"]

    for entry in features:
        assert isinstance(entry, dict)
        assert "name" in entry, f"feature entry {entry} missing 'name'"
        assert "type" in entry, f"feature entry {entry} missing 'type'"
        assert isinstance(entry["name"], str) and entry["name"] != ""
        assert isinstance(entry["type"], str) and entry["type"] != ""


def test_feature_names_are_unique():
    """Sanity check on the checked-in file: feature names shouldn't
    collide with each other (implied by Requirement 1.2's ordered list)."""
    dataset = _load_raw_dataset_dict()
    names = [entry["name"] for entry in dataset["features"]]

    assert len(names) == len(set(names)), "feature names must be unique"


def test_auxiliary_columns_entries_have_name_and_type_when_present():
    """Requirement 1.3: WHERE `auxiliary_columns` are declared, every
    entry has a `name` and a `type`."""
    dataset = _load_raw_dataset_dict()
    aux = dataset.get("auxiliary_columns")

    if aux is None:
        return  # auxiliary_columns is optional; nothing to assert.

    assert isinstance(aux, list)
    for entry in aux:
        assert isinstance(entry, dict)
        assert "name" in entry, f"auxiliary_columns entry {entry} missing 'name'"
        assert "type" in entry, f"auxiliary_columns entry {entry} missing 'type'"
        assert isinstance(entry["name"], str) and entry["name"] != ""
        assert isinstance(entry["type"], str) and entry["type"] != ""


def test_checked_in_file_declares_auxiliary_columns():
    """The checked-in dataset_schema.yaml is known to declare two
    auxiliary columns (fraud_prediction, fraud_probability) -- lock that
    in so a future edit that accidentally drops the section is caught."""
    dataset = _load_raw_dataset_dict()

    assert "auxiliary_columns" in dataset
    assert len(dataset["auxiliary_columns"]) > 0


def test_split_section_declares_strategy_train_ratio_and_hash_column_when_present():
    """Requirement 1.4: WHERE a `split` section is declared, it declares
    `strategy`, `train_ratio`, and `hash_column` values."""
    dataset = _load_raw_dataset_dict()
    split = dataset.get("split")

    if split is None:
        return  # split is optional; nothing to assert.

    assert isinstance(split, dict)
    for key in ("strategy", "train_ratio", "hash_column"):
        assert key in split, f"split section missing '{key}'"

    assert isinstance(split["strategy"], str) and split["strategy"] != ""
    assert isinstance(split["train_ratio"], (int, float))
    assert 0.0 <= float(split["train_ratio"]) <= 1.0
    assert isinstance(split["hash_column"], str) and split["hash_column"] != ""


def test_checked_in_file_declares_split_section():
    """The checked-in dataset_schema.yaml is known to declare a
    deterministic_hash split -- lock that in."""
    dataset = _load_raw_dataset_dict()

    assert "split" in dataset
    assert dataset["split"]["strategy"] == "deterministic_hash"
