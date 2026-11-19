"""Property: the train/validation/test split depends only on row content."""

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from uplift.data import loader
from uplift.data.schema import HILLSTROM
from uplift.data.synthetic import make_experiment

FRAME: pd.DataFrame = make_experiment(n=60, seed=3).frame
BASE = {
    loader.canonical_row_key(row, HILLSTROM.columns): loader.split_of_key(
        loader.canonical_row_key(row, HILLSTROM.columns)
    )
    for _, row in FRAME.iterrows()
}


@given(permutation=st.permutations(list(range(len(FRAME)))))
@settings(max_examples=50, deadline=None)
def test_split_is_deterministic_and_disjoint(permutation: list[int]) -> None:
    shuffled = FRAME.iloc[permutation].reset_index(drop=True)
    labels = loader.row_splits(shuffled)
    keys = [loader.canonical_row_key(row, HILLSTROM.columns) for _, row in shuffled.iterrows()]

    assert len(set(keys)) == len(keys), "fixture rows must be unique for this property"
    for key, label in zip(keys, labels, strict=True):
        assert label == BASE[key]

    for name in ("train", "validation", "test"):
        assert set(labels[labels == name].tolist()) == {name}
    assert set(labels.tolist()) <= {"train", "validation", "test"}
