from collections import namedtuple

import numpy as np

from src.data.prototypes.history_temporal_dataset import TemporalDataset


def parse_split_func_args(
    items: list | np.ndarray | int,
    val_len: int | float | list[int] | np.ndarray,
    test_len: int | float | list[int] | np.ndarray,
) -> tuple[np.ndarray, int | np.ndarray, int | np.ndarray]:
    assert type(val_len) is type(
        test_len
    ), "val_len and test_len must be of the same type."
    if isinstance(items, int):
        items = np.arange(items)
    elif isinstance(items, list):
        items = np.array(items)
    if isinstance(val_len, float):
        assert (
            0 <= val_len <= 1
        ), "Float values must be in the range [0, 1] for percentage splits."
        val_len = int(val_len * len(items))
    if isinstance(test_len, float):
        assert (
            0 <= test_len <= 1
        ), "Float values must be in the range [0, 1] for percentage splits."
        test_len = int(test_len * len(items))
    if isinstance(val_len, list):
        val_len = np.array(val_len)
    if isinstance(test_len, list):
        test_len = np.array(test_len)
    return items, val_len, test_len


def random_split(
    items: list | np.ndarray | int,
    val_len: int | float | list[int] | np.ndarray,
    test_len: int | float | list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    items, val_len, test_len = parse_split_func_args(items, val_len, test_len)
    if isinstance(val_len, np.ndarray):
        val_items = val_len
        test_items = test_len
        train_items = np.setdiff1d(items, np.concatenate([val_items, test_items]))
    else:
        np.random.shuffle(items)
        val_items = items[:val_len]
        test_items = items[val_len : val_len + test_len]
        train_items = items[val_len + test_len :]
    return train_items, val_items, test_items


def sequential_split(
    items: list | np.ndarray | int,
    val_len: int | float | list[int] | np.ndarray,
    test_len: int | float | list[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    items, val_len, test_len = parse_split_func_args(items, val_len, test_len)
    if isinstance(val_len, np.ndarray):
        val_items = val_len
        test_items = test_len
        train_items = np.setdiff1d(items, np.concatenate([val_items, test_items]))
    else:
        test_start = len(items) - test_len
        val_start = test_start - val_len
        train_items = items[:val_start]
        val_items = items[val_start:test_start]
        test_items = items[test_start:]
    return train_items, val_items, test_items


class InductiveBlockSplitter:
    SplitIndices = namedtuple(
        "SplitIndices",
        [
            "train_steps",
            "val_steps",
            "test_steps",
            "train_nodes",
            "val_nodes",
            "test_nodes",
        ],
    )

    def __init__(
        self,
        train_len: list[int] | None = None,
        val_len: int | float | list[int] | np.ndarray | None = None,
        test_len: int | float | list[int] | np.ndarray | None = None,
        val_nodes: int | float | list[int] | np.ndarray | None = None,
        test_nodes: int | float | list[int] | np.ndarray | None = None,
        transductive_only: bool = False,
    ):
        if train_len is not None:
            # explicit range mode
            assert isinstance(train_len, list) and len(train_len) == 2
            assert isinstance(val_len, list) and len(val_len) == 2
            assert isinstance(test_len, list) and len(test_len) == 2
            train_len = (train_len[0], train_len[1])
            val_len = (val_len[0], val_len[1])
            test_len = (test_len[0], test_len[1])
        assert type(val_len) is type(
            test_len
        ), "val_len and test_len must be of the same type."
        assert type(val_nodes) is type(
            test_nodes
        ), "val_nodes and test_nodes must be of the same type."
        super().__init__()
        self._train_len = train_len
        self._val_len = val_len
        self._test_len = test_len
        self._val_nodes = val_nodes
        self._test_nodes = test_nodes
        self._transductive_only = transductive_only
        self._split_indices = None
        self._fitted = False
        self._splits = dict()
        if not self._transductive_only:
            assert self._val_nodes is not None and self._test_nodes is not None

    def __call__(self, *args, **kwargs):
        return self.split(*args, **kwargs)

    def __repr__(self):
        lens = ", ".join(map(lambda kv: "%s=%s" % kv, self.lens().items()))
        return "%s(%s)" % (self.__class__.__name__, lens)

    @property
    def splits(self):
        return self._splits

    def lens(self) -> dict:
        return {k: len(v) for k, v in self.splits.items()}

    @property
    def fitted(self):
        return self._fitted

    def reset(self):
        self._splits = dict()
        self._fitted = False
        self._split_indices = None

    def fit(self, dataset: TemporalDataset) -> dict:
        self._fit_implementation(dataset)
        self._fitted = True
        return self.splits

    def split(self, dataset: TemporalDataset) -> dict:
        if self.fitted:
            return self.splits
        else:
            return self.fit(dataset)

    def _fit_implementation(self, dataset: TemporalDataset):
        # Compute the split indices
        if self._train_len is None:
            train_steps, val_steps, test_steps = sequential_split(
                dataset.n_steps, self._val_len, self._test_len
            )
        else:
            train_steps = np.arange(self._train_len[0], self._train_len[1])
            val_steps = np.arange(self._val_len[0], self._val_len[1])
            test_steps = np.arange(self._test_len[0], self._test_len[1])

        if self._val_nodes is not None and self._test_nodes is not None:
            train_nodes, val_nodes, test_nodes = random_split(
                dataset.n_nodes, self._val_nodes, self._test_nodes
            )
        else:
            train_nodes, val_nodes, test_nodes = None, None, None
        self._split_indices = self.SplitIndices(
            train_steps, val_steps, test_steps, train_nodes, val_nodes, test_nodes
        )

        # Split the dataset
        splits = {
            "train_set": dataset.reduce(time_index=train_steps, node_index=train_nodes),
            "val_set": dataset.reduce(
                time_index=val_steps,
                node_index=train_nodes,
            ),
            "test_set": dataset.reduce(
                time_index=test_steps,
                node_index=train_nodes,
            ),
        }
        # Add re-indexed node indices for each split
        for key in splits:
            splits[key].add_covariate(
                name="node_index",
                value=np.arange(
                    len(train_nodes) if train_nodes is not None else dataset.n_nodes
                ),
                pattern="n",
                add_to_input_map=True,
                preprocess=False,
                convert_precision=False,
            )

        if not self._transductive_only:
            # Additional splits for inductive learning
            splits["val_set_inductive"] = dataset.reduce(
                time_index=val_steps,
                node_index=val_nodes,
            )
            splits["val_set_inductive"].add_covariate(
                name="node_index",
                value=np.arange(len(train_nodes), len(train_nodes) + len(val_nodes)),
                pattern="n",
                add_to_input_map=True,
                preprocess=False,
                convert_precision=False,
            )

            splits["test_set_inductive"] = dataset.reduce(
                time_index=test_steps,
                node_index=test_nodes,
            )
            splits["test_set_inductive"].add_covariate(
                name="node_index",
                value=np.arange(
                    len(train_nodes) + len(val_nodes),
                    len(train_nodes) + len(val_nodes) + len(test_nodes),
                ),
                pattern="n",
                add_to_input_map=True,
                preprocess=False,
                convert_precision=False,
            )

            splits["val_nodes_at_train_time"] = dataset.reduce(
                time_index=train_steps,
                node_index=val_nodes,
            )

            splits["test_nodes_at_train_time"] = dataset.reduce(
                time_index=train_steps,
                node_index=test_nodes,
            )
        self._splits = splits
