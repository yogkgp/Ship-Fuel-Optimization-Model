from ship_fuel.multivessel import repeated_splits
from ship_fuel.preprocessing import clean


def test_repeated_splits_are_ordered_and_embargoed(source):
    data, _ = clean(source)
    settings = {"gap_samples": 12, "repeated_test_starts": [.55, .70, .85],
                "repeated_test_width": .10}
    folds = list(repeated_splits(data, settings))
    assert len(folds) == 3
    for _, train, test in folds:
        assert train["index"].max() < test["index"].min()
        assert test["index"].min() - train["index"].max() >= 12
    assert [fold[2]["index"].min() for fold in folds] == sorted(
        fold[2]["index"].min() for fold in folds)
