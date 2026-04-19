"""
Unit tests for PyTorch Dataset and DataLoader construction.
"""

import numpy as np
import pandas as pd
import pytest

HEMORRHAGE_TYPES = [
    "no_hemorrhage",
    "epidural",
    "intraparenchymal",
    "intraventricular",
    "subarachnoid",
    "subdural",
]


def _make_fake_df(n: int = 8) -> pd.DataFrame:
    """Build a small DataFrame with random labels but no real image paths."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        row = {"image_path": f"/fake/scan_{i:03d}.png"}
        labels = (rng.random(len(HEMORRHAGE_TYPES)) > 0.7).astype(int)
        for col, val in zip(HEMORRHAGE_TYPES, labels):
            row[col] = int(val)
        rows.append(row)
    return pd.DataFrame(rows)


class TestICHDataset:
    def test_len(self):
        from src.data.dataset import ICHDataset
        df = _make_fake_df(10)
        ds = ICHDataset(df)
        assert len(ds) == 10

    def test_label_shape(self):
        from src.data.dataset import ICHDataset
        df = _make_fake_df(4)
        ds = ICHDataset(df)
        assert ds.labels.shape == (4, len(HEMORRHAGE_TYPES))

    def test_label_dtype(self):
        from src.data.dataset import ICHDataset
        df = _make_fake_df(4)
        ds = ICHDataset(df)
        assert ds.labels.dtype == np.float32
