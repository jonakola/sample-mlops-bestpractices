"""Unit tests for Schema_Loader's error handling and process-lifetime caching.

Covers three behaviors of `src/config/schema.py`'s internal `_load()`:

1. A missing `dataset_schema.yaml` raises `FileNotFoundError` naming the
   resolved path Schema_Loader expected to find it at.
2. A parsed document that omits the top-level `dataset` key raises
   `ValueError` naming that missing key.
3. `_load()` is decorated with `functools.lru_cache(maxsize=1)`, so calling
   any accessor more than once in the same process parses the YAML file
   only once.

`_load()`'s `functools.lru_cache` is a module-global cache shared with every
other test file that imports `src.config.schema`. An autouse fixture clears
it before and after every test in this file so these tests neither read a
stale cached value left by another test file nor leave a patched
path/result cached for tests that run afterward.

**Validates: Requirements 2.17, 2.18, 2.19**
"""

import pytest
import yaml as real_yaml

from src.config import schema


@pytest.fixture(autouse=True)
def _clear_schema_cache():
    """Ensure `_load()`'s cache starts and ends empty for every test here.

    Without this, a real (uncached) call from another test module could
    populate the cache before these tests run (masking the FileNotFoundError/
    ValueError cases), and a patched result left behind by one of these tests
    could leak into whichever test runs next.
    """
    schema._load.cache_clear()
    yield
    schema._load.cache_clear()


def _write_yaml(tmp_path, content: str):
    path = tmp_path / "dataset_schema.yaml"
    path.write_text(content)
    return path


def test_missing_schema_file_raises_file_not_found_naming_resolved_path(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist" / "dataset_schema.yaml"
    monkeypatch.setattr(schema, "_SCHEMA_YAML_PATH", missing_path)

    with pytest.raises(FileNotFoundError) as exc_info:
        schema.identifier_column()

    assert str(missing_path) in str(exc_info.value)


def test_missing_dataset_key_raises_value_error_naming_key(tmp_path, monkeypatch):
    path = _write_yaml(tmp_path, "not_dataset:\n  foo: bar\n")
    monkeypatch.setattr(schema, "_SCHEMA_YAML_PATH", path)

    with pytest.raises(ValueError) as exc_info:
        schema.identifier_column()

    assert "dataset" in str(exc_info.value)


class _CountingYaml:
    """Wraps the real `yaml` module, counting `safe_load()` calls.

    Bound onto `schema.yaml` (not the global `yaml` module) via monkeypatch,
    so the call-count tracking is scoped to this test and never leaks into
    other tests that import `yaml` directly.
    """

    def __init__(self, real_module):
        self._real_module = real_module
        self.safe_load_call_count = 0

    def safe_load(self, stream):
        self.safe_load_call_count += 1
        return self._real_module.safe_load(stream)


def test_calling_accessor_twice_parses_file_only_once(tmp_path, monkeypatch):
    path = _write_yaml(
        tmp_path,
        "dataset:\n"
        "  identifier_column: id\n"
        "  timestamp_column: ts\n"
        "  target_column: target\n"
        "  features:\n"
        "    - name: f1\n"
        "      type: double\n",
    )
    monkeypatch.setattr(schema, "_SCHEMA_YAML_PATH", path)

    counting_yaml = _CountingYaml(real_yaml)
    monkeypatch.setattr(schema, "yaml", counting_yaml)

    # Two different accessors, both routed through the cached `_load()`.
    assert schema.identifier_column() == "id"
    assert schema.feature_names() == ["f1"]

    assert counting_yaml.safe_load_call_count == 1
