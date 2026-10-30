from contextlib import contextmanager
from copy import deepcopy
from typing import Callable, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from pandas import DatetimeIndex
from torch import Tensor
from torch.utils.data import Dataset
from tsl.data import BatchMap, Data, BatchMapItem, SynchMode, StaticBatch
from tsl.data.preprocessing import Scaler, ScalerModule
from tsl.ops.pattern import outer_pattern, check_pattern, infer_pattern
from tsl.typing import DataArray, IndexSlice, TemporalIndex
from tsl.utils import casting
from tsl.utils.casting import parse_index

_WINDOWING_ = {
    "window": ["window", "window_lag"],
    "horizon": ["window", "delay", "horizon", "horizon_lag"],
    "history": ["history"],
    "indices": [
        "target",
        "window",
        "delay",
        "horizon",
        "stride",
        "history",
        "history_shift",
    ],
    "all": [
        "target",
        "window",
        "delay",
        "horizon",
        "stride",
        "horizon_lag",
        "window_lag",
        "history",
        "history_shift",
    ],
}


class TemporalDataParsingMixin:
    def _parse_target(self, obj: DataArray) -> Tensor:
        obj = casting.copy_to_tensor(obj)
        obj = casting.to_time_nodes_channels(obj)
        obj = casting.convert_precision_tensor(obj, precision=self.precision)
        return obj

    def _parse_covariate(
        self,
        obj: DataArray,
        pattern: str | None = None,
        allow_broadcasting: bool = False,
        convert_precision: bool = True,
        name: str = "covariate",
    ) -> tuple[Tensor, str]:
        # convert to tensor
        obj = casting.copy_to_tensor(obj)

        # infer pattern if it is None, otherwise sanity check
        if pattern is None:
            pattern = infer_pattern(obj.shape, t=self.n_steps, n=self.n_nodes)
        else:
            pattern = check_pattern(pattern, ndim=obj.ndim)

        # check that pattern and shape match
        self._check_pattern(obj, pattern, name, allow_broadcasting=allow_broadcasting)

        if convert_precision:
            obj = casting.convert_precision_tensor(obj, self.precision)

        return obj, pattern

    def _check_pattern(
        self, obj: Tensor, pattern: str, name: str, allow_broadcasting: bool = False
    ):
        dims = pattern.strip().split(" ")
        for token, size in zip(dims, obj.size()):
            if token == "t":
                self._check_same_dim(size, "n_steps", name, allow_broadcasting)
            elif token == "n":
                self._check_same_dim(size, "n_nodes", name, allow_broadcasting)

    def _check_same_dim(
        self, dim: int, attr: str, name: str, allow_broadcasting: bool = False
    ):
        dim_data = getattr(self, attr)
        if not (dim == dim_data or (dim == 1 and allow_broadcasting)):
            raise ValueError(
                "Cannot assign {0} with {1}={2}: data has {1}={3}".format(
                    name, attr, dim, dim_data
                )
            )

    def _check_name(self, name: str):
        # name cannot be an attribute of self, nor a key in get
        invalid_names = set(dir(self))
        if name in invalid_names:
            raise ValueError(
                f"Cannot set attribute with name '{name}', there "
                f"is already an attribute named '{name}' in the "
                "dataset."
            )

    @staticmethod
    def _value_to_kwargs(value: DataArray | list | tuple | Mapping):
        keys = [
            "value",
            "pattern",
            "add_to_input_map",
            "synch_mode",
            "preprocess",
            "convert_precision",
        ]
        if isinstance(value, (pd.DataFrame, np.ndarray, Tensor)):
            return dict(value=value)
        if isinstance(value, (list, tuple)):
            return dict(zip(keys, value))
        elif isinstance(value, Mapping):
            assert set(value.keys()).issubset(keys)
            return value
        else:
            raise TypeError('Invalid type for value "{}"'.format(type(value)))


class TemporalDataset(Dataset, TemporalDataParsingMixin):
    r"""Base class for structures that are bridges between Datasets and Models.

    A :class:`SpatioTemporalDataset` takes as input a
    :class:`~tsl.datasets.prototypes.Dataset` and
    build a proper structure to feed deep models.

    Args:
        target (DataArray): Data relative to the primary channels.
        index (TemporalIndex, optional): Temporal indices for the data.
            (default: :obj:`None`)
        mask (DataArray, optional): Boolean mask denoting if signal in data is
            valid (1) or not (0).
            (default: :obj:`None`)
        covariates (dict, optional): Dictionary of exogenous channels with
            label. An :obj:`exogenous` element is a temporal array with node- or
            graph-level channels which are covariates to the main signal. The
            temporal dimension must be equal to the temporal dimension of data,
            as well as the number of nodes if the exogenous is node-level.
            (default: :obj:`None`)
        input_map (BatchMap or dict, optional): Defines how data (i.e., the
            target and the covariates) are mapped to dataset sample input. Keys
            in the mapping are keys in both :obj:`item` and :obj:`item.input`,
            while values are :obj:`~tsl.data.new.BatchMapItem`.
            (default: :obj:`None`)
        target_map (BatchMap or dict, optional): Defines how data (i.e., the
            target and the covariates) are mapped to dataset sample target. Keys
            in the mapping are keys in both :obj:`item` and :obj:`item.target`,
            while values are :obj:`~tsl.data.new.BatchMapItem`.
            (default: :obj:`None`)
        auxiliary_map (BatchMap or dict, optional): Defines how data (i.e., the
            target and the covariates) are added as additional attributes to the
            dataset sample. Keys in the mapping are keys only in :obj:`item`,
            while values are :obj:`~tsl.data.new.BatchMapItem`.
            (default: :obj:`None`)
        scalers (Mapping or None): Dictionary of scalers that must be used for
            data preprocessing.
            (default: :obj:`None`)
        trend (DataArray, optional): Trend paired with main signal. Must be of
            the same shape of `data`.
            (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in a
            :class:`tsl.data.Data` object and returns a transformed version.
            The data object will be transformed before every access.
            (default: :obj:`None`)
        window (int): Length (in number of steps) of the lookback window.
            (default: 12)
        horizon (int): Length (in number of steps) of the prediction horizon.
            (default: 1)
        delay (int): Offset (in number of steps) between end of window and start
            of horizon.
            (default: 0)
        stride (int): Offset (in number of steps) between a sample and the next
            one.
            (default: 1)
        window_lag (int): Sampling frequency (in number of steps) in lookback
            window.
            (default: 1)
        horizon_lag (int): Sampling frequency (in number of steps) in prediction
            horizon.
            (default: 1)
        precision (int or str, optional): The float precision to store the data.
            Can be expressed as number (16, 32, or 64) or string ("half",
            "full", "double").
            (default: 32)
        name (str, optional): The (optional) name of the dataset.
        history (int): Maximum number of historical steps to
            consider in the sample.
            (default: 0)
        history_shift (int or tuple): If int, the number of steps to
            shift the history indices. If tuple, a random number of steps will be
            sampled from the range defined by the tuple.
            (default: 0)
    """

    def __init__(
        self,
        target: DataArray,
        index: TemporalIndex | None = None,
        mask: DataArray | None = None,
        covariates: Mapping[str, DataArray] | None = None,
        input_map: Mapping | BatchMap | None = None,
        target_map: Mapping | BatchMap | None = None,
        auxiliary_map: Mapping | BatchMap | None = None,
        scalers: Mapping[str, Scaler] | None = None,
        trend: DataArray | None = None,
        transform: Callable | None = None,
        window: int = 12,
        horizon: int = 1,
        delay: int = 0,
        stride: int = 1,
        window_lag: int = 1,
        horizon_lag: int = 1,
        precision: int | str = 32,
        name: str | None = None,
        history: int = 0,
        history_shift: int | tuple[int, int] = 0,
        history_shift_mode: Literal["random", "upto", "fixed"] = "random",
    ):
        super().__init__()

        # Set info
        self.name = name if name is not None else self.__class__.__name__
        self.precision = precision

        # Store windowing information
        self.window = window
        self.delay = delay
        self.horizon = horizon
        self.stride = stride
        self.window_lag = window_lag
        self.horizon_lag = horizon_lag

        # Initialize private data holders
        self._covariates = dict()
        self._indices = None

        # Initialize batch maps
        self.input_map = BatchMap()
        self.target_map = BatchMap()
        self.auxiliary_map = BatchMap()

        # Store preprocessing modules
        self.scalers: dict = dict()
        self.trend: Tensor | None = None
        self.transform = transform
        # cache scalers for composed batch items
        self._batch_scalers: dict = dict()
        # handle trend as bias in target scaler, so cache actual one (if any)
        # to restore after trend update/deletion
        self.__target_bias: Tensor | None = None

        # Store time information
        # if index is not explicitly passed and data is a dataframe, defaults to
        # data's index
        if index is None and isinstance(target, pd.DataFrame):
            if isinstance(target.index, DatetimeIndex):
                index = target.index
        self.index = index

        # Set dataset's target signals
        self.target: Tensor = self._parse_target(target)

        # Store mask
        self.mask: Tensor | None = None
        self.set_mask(mask)

        # Store covariates (e.g., exogenous and attributes)
        self.reset_input_map()
        if covariates is not None:
            for name, value in covariates.items():
                self.add_covariate(name, **self._value_to_kwargs(value))

        # Updated input map (i.e., how to map data, exogenous and attribute
        # inside item.input)
        if input_map is not None:
            self.set_input_map(input_map)

        # Updated target map (i.e., how to map data, exogenous and attribute
        # inside item.target)
        self.reset_target_map()
        if target_map is not None:
            self.set_target_map(target_map)

        # Updated auxiliary map (i.e., how to map data, exogenous and attribute
        # inside item)
        if auxiliary_map is not None:
            self.set_auxiliary_map(auxiliary_map)

        # A scaler is a module that transforms data with a linear operation
        if scalers is not None:
            for k, v in scalers.items():
                self.add_scaler(k, v)

        # Target's trend (i.e., 't n f' tensor to be removed when target
        # is preprocessed)
        if trend is not None:
            self.set_trend(trend)

        # Set history
        self.history = history
        self.history_shift = history_shift
        self.history_shift_mode = history_shift_mode

    def __repr__(self):
        return "{}(n_samples={}, n_nodes={}, n_channels={})".format(
            self.name, self.n_samples, self.n_nodes, self.n_channels
        )

    def __getitem__(self, item) -> Data:
        if isinstance(item, int) and item < 0:
            # compute item's actual index
            item = self._indices.size(0) + item
        else:
            # convert slice to indexes
            item = parse_index(item, length=self.n_samples, layout="index")
        item = self.get(item)
        if self.transform is not None:
            item = self.transform(item)
        return item

    def __contains__(self, item) -> bool:
        return item in self.keys

    def __len__(self) -> int:
        return self.n_samples

    def __getattr__(self, item):
        if "_covariates" in self.__dict__ and item in self._covariates:
            return self._covariates[item]["value"]
        if "input_map" in self.__dict__ and item in self.input_map:
            return self.collate_item_elem(item)[0]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no " f"attribute '{item}'"
        )

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key in _WINDOWING_["window"] and all(
            [hasattr(self, attr) for attr in _WINDOWING_["window"]]
        ):
            self._window_range = torch.arange(0, self.window, self.window_lag)

        if key in _WINDOWING_["horizon"] and all(
            [hasattr(self, attr) for attr in _WINDOWING_["horizon"]]
        ):
            self._horizon_range = torch.arange(
                self.horizon_offset,
                self.horizon_offset + self.horizon,
                self.horizon_lag,
            )

        if key in _WINDOWING_["history"] and all(
            [hasattr(self, attr) for attr in _WINDOWING_["history"]]
        ):
            if self.history > 0:
                self._history_range = torch.arange(-self.history, 0, self.window_lag)
            else:
                self._history_range = None

        if key in _WINDOWING_["indices"] and all(
            [hasattr(self, attr) for attr in _WINDOWING_["indices"]]
        ):
            start = 0 if self.history == 0 else self.history + self.history_shift[0]
            self._indices = torch.cartesian_prod(
                torch.arange(start, self.n_steps - self.sample_span + 1, self.stride),
                torch.arange(0, self.n_nodes),
            )

    def __delattr__(self, item):
        if item in _WINDOWING_["all"]:
            raise AttributeError(f"Cannot delete attribute '{item}'.")
        elif item == "mask":
            self.set_mask(None)
        elif item == "trend":
            self.set_trend(None)
        elif item in self._covariates:
            self.remove_covariate(item)
        else:
            super().__delattr__(item)

    # Dataset properties ######################################################

    @property
    def n_steps(self) -> int:
        """Total number of time steps in the dataset."""
        return self.target.size(0)

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the dataset."""
        return self.target.size(1)

    @property
    def n_channels(self) -> int:
        """Number of channels in dataset's target."""
        return self.target.size(-1)

    @property
    def shape(self) -> tuple:
        """Shape of the target tensor."""
        return tuple(self.target.size())

    @property
    def patterns(self) -> dict:
        """Shows the dimension of dataset's tensors in a more informative way.

        The pattern mapping can be useful to glimpse on how data are arranged.
        The convention we use is the following:

        * 't' stands for “number of time steps”
        * 'n' stands for “number of nodes”
        * 'f' stands for “number of features” (per node)
        """
        patterns = {"target": "t n f"}
        # add mask pattern
        if self.mask is not None:
            patterns["mask"] = "t n f"
        # add covariates patterns
        patterns.update(
            {name: attr["pattern"] for name, attr in self._covariates.items()}
        )
        return patterns

    @property
    def batch_patterns(self) -> dict:
        """Shows the dimension of dataset's tensors in a more informative way.

        The pattern mapping can be useful to glimpse on how data are arranged.
        The convention we use is the following:

        * 't' stands for “number of time steps”
        * 'n' stands for “number of nodes”
        * 'f' stands for “number of features” (per node)
        """
        # add input map patterns
        patterns = {name: attr.pattern for name, attr in self.input_map.items()}
        # add mask pattern
        if self.mask is not None:
            patterns["mask"] = "t n f"
        # add target map patterns
        patterns.update({name: attr.pattern for name, attr in self.target_map.items()})
        # add auxiliary map patterns
        patterns.update(
            {name: attr.pattern for name, attr in self.auxiliary_map.items()}
        )
        return patterns

    @property
    def keys(self) -> list:
        """Keys in dataset."""
        return list(self.patterns.keys())

    @property
    def batch_keys(self) -> list:
        """Keys in dataset item."""
        return list(self.batch_patterns.keys())

    # Indexing

    @property
    def history(self):
        """Maximum number of historical steps to consider in the sample."""
        return self._history

    @history.setter
    def history(self, value: int):
        """Set maximum number of historical steps to consider in the sample."""
        assert isinstance(value, int) and (value >= 0)
        self._history = value

    @property
    def history_shift(self) -> tuple[int, int]:
        return self._history_shift

    @history_shift.setter
    def history_shift(self, value: int | tuple[int, int]):
        """Set the number of steps to shift the history indices."""
        if isinstance(value, int):
            assert value >= 0
            value = (0, value)
        elif isinstance(value, Sequence):
            assert (
                len(value) == 2
                and value[1] >= value[0] >= 0
                and isinstance(value[0], int)
                and isinstance(value[1], int)
            )
        else:
            raise AttributeError("History shift is neither a tuple nor an integer.")
        self._history_shift = value

    @property
    def history_shift_mode(self) -> Literal["random", "upto", "fixed"]:
        """Mode of history shift."""
        return self._history_shift_mode

    @history_shift_mode.setter
    def history_shift_mode(self, value: Literal["random", "upto", "fixed"]):
        """Set the mode of history shift."""
        if value not in ["random", "upto", "fixed"]:
            raise ValueError(
                f"Invalid history shift mode: {value}. "
                "Must be one of 'random', 'upto', or 'fixed'."
            )
        self._history_shift_mode = value

    @property
    def horizon_offset(self) -> int:
        """Lag of starting step of horizon."""
        return self.window + self.delay

    @property
    def sample_span(self) -> int:
        """Total number of steps of an item, including window and horizon."""
        return max(self.horizon_offset + self.horizon, self.window)

    @property
    def samples_offset(self) -> int:
        """Difference (in number of steps) between two items."""
        return int(np.ceil(self.window / self.stride))

    @property
    def indices(self) -> Tensor:
        """Indices of the dataset. The :obj:`i`-th item is mapped to
        :obj:`indices[i]`"""
        return self._indices

    @property
    def n_samples(self) -> int:
        """Number of samples (i.e., items) in the dataset."""
        return len(self._indices)

    # Covariates properties

    @property
    def covariates(self) -> dict:
        """Mapping of dataset's covariates."""
        return {name: attr["value"] for name, attr in self._covariates.items()}

    @property
    def exogenous(self) -> dict:
        """Time-varying covariates of the dataset's target."""
        return {
            name: attr["value"]
            for name, attr in self._covariates.items()
            if "t" in attr["pattern"]
        }

    @property
    def attributes(self) -> dict:
        """Static features related to the dataset."""
        return {
            name: attr["value"]
            for name, attr in self._covariates.items()
            if "t" not in attr["pattern"]
        }

    @property
    def n_covariates(self) -> int:
        """Number of covariates in the dataset."""
        return len(self._covariates)

    # flags

    @property
    def has_mask(self) -> bool:
        """Whether the dataset has a mask denoting valid values in target."""
        return self.mask is not None

    @property
    def has_covariates(self) -> bool:
        """Whether the dataset has covariates to the target tensor."""
        return self.n_covariates > 0

    # Map Dataset to item #####################################################

    @property
    def targets(self) -> BatchMap:
        return self.target_map

    def reset_input_map(self):
        self._clear_batch_map("input")
        self.input_map["x"] = BatchMapItem(
            "target",
            SynchMode.WINDOW,
            preprocess=True,
            cat_dim=None,
            pattern="t n f",
            shape=self.shape,
        )
        for name, attr in self._covariates.items():
            self.input_map[name] = BatchMapItem(
                name,
                SynchMode.WINDOW,
                preprocess=True,
                cat_dim=None,
                pattern=attr["pattern"],
                shape=attr["value"].size(),
            )

    def reset_target_map(self):
        self._clear_batch_map("target")
        self.target_map["y"] = BatchMapItem(
            "target",
            SynchMode.HORIZON,
            preprocess=False,
            cat_dim=None,
            pattern="t n f",
            shape=self.shape,
        )

    def reset_auxiliary_map(self):
        self._clear_batch_map("auxiliary")
        if self.mask is not None:
            self.auxiliary_map["mask"] = BatchMapItem(
                "mask",
                SynchMode.HORIZON,
                preprocess=False,
                cat_dim=None,
                pattern="t n f",
                shape=self.mask.shape,
            )

    def set_input_map(self, input_map=None, **kwargs):
        self._clear_batch_map("input")
        self.update_input_map(input_map, **kwargs)

    def set_target_map(self, target_map=None, **kwargs):
        self._clear_batch_map("target")
        self.update_target_map(target_map, **kwargs)

    def set_auxiliary_map(self, auxiliary_map=None, **kwargs):
        self._clear_batch_map("auxiliary")
        self.update_auxiliary_map(auxiliary_map, **kwargs)

    def update_input_map(self, input_map=None, **kwargs):
        self._update_batch_map("input", input_map, **kwargs)

    def update_target_map(self, target_map=None, **kwargs):
        self._update_batch_map("target", target_map, **kwargs)

    def update_auxiliary_map(self, auxiliary_map=None, **kwargs):
        self._update_batch_map("auxiliary", auxiliary_map, **kwargs)

    def _clear_batch_map(self, endpoint):
        batch_map = getattr(self, f"{endpoint}_map")
        for key in batch_map:
            if key in self._batch_scalers:
                del self._batch_scalers[key]
        setattr(self, f"{endpoint}_map", BatchMap())

    def _update_batch_map(self, endpoint, batch_map=None, **kwargs):
        # check batch map
        if batch_map is None:
            batch_map = BatchMap()
        elif not isinstance(batch_map, Mapping):
            raise TypeError(
                f"Type {type(batch_map)} is not valid for "
                f"`{endpoint}_map` (must be a mapping)."
            )
        # update from kwargs
        batch_map.update(**kwargs)
        # update endpoint_map
        endpoint_map = getattr(self, f"{endpoint}_map")
        endpoint_map.update(**batch_map)
        # update 'pattern' and 'shape' to added/updated keys
        for name in batch_map:
            item: BatchMapItem = endpoint_map[name]
            # keys sanity check and compute pattern and shape
            keys = item.keys
            if len(keys) > 1:
                tensor, scaler, pattern = self.collate_keys(
                    keys, cat_dim=item.cat_dim, return_pattern=True
                )
                item.shape, item.pattern = tuple(tensor.size()), pattern
                if scaler is not None:
                    self._batch_scalers[name] = scaler
            else:
                item.shape = tuple(getattr(self, keys[0]).size())
                item.pattern = self.patterns[keys[0]]

    # Getters #################################################################

    def get(self, item):
        # check if item is scalar or vector
        ndim = item.ndim if isinstance(item, Tensor) else 0
        if ndim == 0:  # get a single item
            sample = Data(pattern=self.batch_patterns)
        elif ndim == 1:  # get batch of items
            # batch dim is always added
            pattern = {
                name: ("b " + pattern) for name, pattern in self.batch_patterns.items()
            }
            sample = StaticBatch(pattern=pattern, size=item.size(0))
        else:
            raise RuntimeError(f"Too many dimensions for index ({ndim}).")

        # get indices
        t_idxs, n_idxs = self.get_item_indices(item)

        # get input synchronized with window
        if "window" in t_idxs:
            wdw_t_idxs = t_idxs["window"]
            self._add_to_sample(
                sample,
                SynchMode.WINDOW,
                "input",
                time_index=wdw_t_idxs,
                node_index=n_idxs,
            )
            self._add_to_sample(
                sample,
                SynchMode.WINDOW,
                "target",
                time_index=wdw_t_idxs,
                node_index=n_idxs,
            )
            self._add_to_sample(
                sample,
                SynchMode.WINDOW,
                "auxiliary",
                time_index=wdw_t_idxs,
                node_index=n_idxs,
            )

        # get input synchronized with horizon
        hrz_t_idxs = t_idxs["horizon"]
        self._add_to_sample(
            sample,
            SynchMode.HORIZON,
            "input",
            time_index=hrz_t_idxs,
            node_index=n_idxs,
        )
        self._add_to_sample(
            sample,
            SynchMode.HORIZON,
            "target",
            time_index=hrz_t_idxs,
            node_index=n_idxs,
        )
        self._add_to_sample(
            sample,
            SynchMode.HORIZON,
            "auxiliary",
            time_index=hrz_t_idxs,
            node_index=n_idxs,
        )

        # get static data
        self._add_to_sample(sample, SynchMode.STATIC, "input", node_index=n_idxs)
        self._add_to_sample(sample, SynchMode.STATIC, "target", node_index=n_idxs)
        self._add_to_sample(sample, SynchMode.STATIC, "auxiliary", node_index=n_idxs)

        # get input synchronized with history
        if self.history > 0:
            hst_t_idxs = t_idxs["history"]
            self._add_to_sample(
                sample,
                SynchMode.HISTORY,
                "input",
                time_index=hst_t_idxs,
                node_index=n_idxs,
            )
            self._add_to_sample(
                sample,
                SynchMode.HISTORY,
                "target",
                time_index=hst_t_idxs,
                node_index=n_idxs,
            )
            self._add_to_sample(
                sample,
                SynchMode.HISTORY,
                "auxiliary",
                time_index=hst_t_idxs,
                node_index=n_idxs,
            )
        return sample

    def expand_scaler(
        self,
        key: str,
        pattern: str | None = None,
        time_index: list | Tensor | None = None,
        node_index: list | Tensor | None = None,
    ) -> ScalerModule | None:
        # check if there is a scaler
        if key not in self.keys:
            raise KeyError(f"{key} not in {self.name}.")
        elif key not in self.scalers:
            return None
        # get scaler
        if pattern is None:
            return self.scalers[key]
        # convert indices
        time_index = self._get_time_index(time_index, layout="index")
        node_index = self._get_node_index(node_index, layout="index")
        # if there is an out-pattern, create new scaler
        scaler = self.scalers[key]
        (bias, scale), out_pattern = take(
            [scaler.bias, scaler.scale],
            in_pattern=self.patterns[key],
            out_pattern=pattern,
            time_index=time_index,
            node_index=node_index,
            try_index_1d=False,
        )
        scaler = ScalerModule(scaler, pattern=out_pattern)
        scaler.bias = bias
        scaler.scale = scale
        return scaler

    def expand_tensor(
        self,
        key: str,
        pattern: str,
        time_index: list | Tensor | None = None,
        node_index: list | Tensor | None = None,
    ) -> Tensor:
        # convert indices
        time_index = self._get_time_index(time_index, layout="index")
        node_index = self._get_node_index(node_index, layout="index")
        x, out_pattern = take(
            getattr(self, key),
            in_pattern=self.patterns[key],
            out_pattern=pattern,
            time_index=time_index,
            node_index=node_index,
            t=self.n_steps,
            n=self.n_nodes,
        )
        return x

    def get_tensor(
        self,
        key: str,
        preprocess: bool = False,
        time_index: list | Tensor | None = None,
        node_index: list | Tensor | None = None,
    ) -> tuple[Tensor, ScalerModule | None]:
        # get dataset item
        if key not in self.keys:
            raise KeyError(f"{key} not in dataset {self.name}.")

        # convert indices
        time_index = self._get_time_index(time_index, layout="index")
        node_index = self._get_node_index(node_index, layout="index")
        x, x_pattern = take(
            getattr(self, key),
            in_pattern=self.patterns[key],
            out_pattern=None,
            time_index=time_index,
            node_index=node_index,
        )

        # get scaler (if any), assumes bias and scale have same shape
        scaler = None
        if key in self.scalers is not None:
            scaler = self.scalers[key]
            if scaler.pattern is None:
                raise RuntimeError("You are trying to index a scaler with no pattern.")
            (bias, scale), scaler_pattern = take(
                [scaler.bias, scaler.scale],
                in_pattern=scaler.pattern,
                out_pattern=None,
                time_index=time_index,
                node_index=node_index,
                try_index_1d=False,
            )
            if x_pattern.startswith("b") and not scaler_pattern.startswith("b"):
                bias = bias.unsqueeze(0)
                scale = scale.unsqueeze(0)
                scaler_pattern = "b " + scaler_pattern
            # create scaler module
            scaler = ScalerModule(
                scaler=scaler,
                pattern=scaler_pattern,
            )
            scaler.bias = bias
            scaler.scale = scale
            if preprocess:  # transform tensor
                x = scaler.transform(x)
        return x, scaler

    def collate_item_elem(
        self,
        key: str,
        time_index: list | Tensor | None = None,
        node_index: list | Tensor | None = None,
    ) -> tuple[Tensor, ScalerModule | None]:
        # get batch item
        if key in self.input_map:
            itm = self.input_map[key]
        elif key in self.target_map:
            itm = self.target_map[key]
        # FIX: Added auxiliary_map to the condition, missing in TSL
        elif key in self.auxiliary_map:
            itm = self.auxiliary_map[key]
        # END_FIX
        else:
            raise KeyError(f"{key} not in any batch map of {self.name}.")

        # expand and concatenate tensors
        x = torch.cat(
            [
                self.expand_tensor(k, itm.pattern, time_index, node_index)
                for k in itm.keys
            ],
            dim=itm.cat_dim,
        )
        # get scaler (if any)
        scaler = None
        if key in self._batch_scalers:
            scaler = self._batch_scalers[key]
            if scaler.pattern is None:
                raise RuntimeError("You are trying to index a scaler with no pattern.")
            (bias, scale), scaler_pattern = take(
                [scaler.bias, scaler.scale],
                in_pattern=scaler.pattern,
                out_pattern=None,
                time_index=time_index,
                node_index=node_index,
                try_index_1d=False,
            )
            if x.ndim > bias.ndim:  # check if there is an additional batch dimension
                bias = bias.unsqueeze(0)
                scale = scale.unsqueeze(0)
                scaler_pattern = "b " + scaler_pattern
            # create scaler module
            scaler = ScalerModule(
                scaler=scaler,
                pattern=scaler_pattern,
            )
            scaler.bias = bias
            scaler.scale = scale
            if itm.preprocess:  # transform tensor
                x = scaler.transform(x)
        return x, scaler

    def collate_keys(
        self,
        keys: Iterable,
        preprocess: bool = False,
        time_index: list | Tensor | None = None,
        node_index: list | Tensor | None = None,
        cat_dim: int | None = None,
        return_pattern: bool = False,
    ):
        if any([key not in self.keys for key in keys]):
            unmatch = set(keys).difference(self.keys)
            raise KeyError(f"{unmatch} not in {self.name}.")
        pattern = outer_pattern([self.patterns[key] for key in keys])
        tensors, scalers = list(), list()
        for key in keys:
            tensor = self.expand_tensor(key, pattern, time_index, node_index)
            scaler = self.expand_scaler(key, pattern, time_index, node_index)
            if preprocess and scaler is not None:
                tensor = scaler(tensor)
            tensors.append(tensor)
            scalers.append(scaler)
        if len(tensors) == 1:
            if return_pattern:
                return tensors[0], scalers[0], pattern
            return tensors[0], scalers[0]
        if cat_dim is not None:
            scalers = ScalerModule.cat(
                scalers, dim=cat_dim, sizes=[t.size() for t in tensors]
            )
            tensors = torch.cat(tensors, dim=cat_dim)
        if return_pattern:
            return tensors, scalers, pattern
        return tensors, scalers

    def get_mask(self, dtype: type | str | np.dtype | None = None) -> Tensor:
        mask = self.mask if self.has_mask else ~torch.isnan(self.target)
        if dtype is not None:
            assert dtype in ["bool", "uint8", bool, torch.bool, torch.uint8]
            mask = mask.to(dtype)
        return mask

    # Getters helpers

    def _get_time_index(
        self,
        time_index: IndexSlice = None,
        layout: Literal["index", "slice", "mask"] = "index",
    ):
        return parse_index(time_index, length=self.n_steps, layout=layout)

    def _get_node_index(
        self,
        node_index: IndexSlice = None,
        layout: Literal["index", "slice", "mask"] = "index",
    ):
        return parse_index(node_index, length=self.n_nodes, layout=layout)

    def _add_to_sample(
        self, out, synch_mode, endpoint="input", time_index=None, node_index=None
    ):
        batch_map = getattr(self, f"{endpoint}_map")
        for key, item in batch_map.by_synch_mode(synch_mode).items():
            if len(item.keys) > 1:
                tensor, scaler = self.collate_item_elem(
                    key, time_index=time_index, node_index=node_index
                )
            else:
                tensor, scaler = self.get_tensor(
                    item.keys[0],
                    preprocess=item.preprocess,
                    time_index=time_index,
                    node_index=node_index,
                )
            if not isinstance(out, StaticBatch):
                # if out is not a StaticBatch then remove batch dimension
                tensor = tensor.squeeze(0)
                if scaler is not None:
                    scaler = ScalerModule(
                        scaler=scaler,
                        pattern=scaler.pattern[2:],  # remove 'b ' prefix,
                    )
                    scaler.bias = scaler.bias.squeeze(0)
                    scaler.scale = scaler.scale.squeeze(0)
            # add to output
            if endpoint == "auxiliary":
                out[key] = tensor
            else:
                getattr(out, endpoint)[key] = tensor
            if scaler is not None:
                out.transform[key] = scaler

    # Setters #################################################################

    def set_data(self, data: DataArray):
        r"""Set sequence of primary channels at :obj:`self.data`."""
        self.target = self._parse_target(data)

    def set_mask(self, mask: DataArray | None, add_to_auxiliary_map: bool = True):
        r"""Set mask of target channels, i.e., a bool for each (node, time
        step, channel) triplet denoting if corresponding value in target is
        observed (obj:`True`) or not (obj:`False`)."""
        if mask is not None:
            mask = self._parse_target(mask)
            if mask.dtype not in [torch.bool, torch.uint8]:
                raise RuntimeError(
                    "Mask is not boolean, accepted types are "
                    f"{[torch.bool, torch.uint8]}."
                )
            # check mask length is equal to target's length
            self._check_same_dim(mask.size(0), "n_steps", "mask")
            # check mask nodes/features are broadcastable to
            # target's nodes/features
            self._check_same_dim(
                mask.size(1), "n_nodes", "mask", allow_broadcasting=True
            )
            self._check_same_dim(
                mask.size(-1), "n_channels", "mask", allow_broadcasting=True
            )
            if add_to_auxiliary_map:
                self.auxiliary_map["mask"] = BatchMapItem(
                    "mask",
                    SynchMode.HORIZON,
                    preprocess=False,
                    cat_dim=None,
                    pattern="t n f",
                    shape=mask.shape,
                )
        self.mask = mask

    def add_covariate(
        self,
        name: str,
        value: DataArray,
        pattern: str | None = None,
        add_to_input_map: bool = True,
        synch_mode: SynchMode | None = None,
        preprocess: bool = True,
        convert_precision: bool = True,
    ):
        r"""Add covariate to the dataset. Examples of covariate are
        exogenous signals (in the form of dynamic multidimensional data) or
        static attributes (e.g., graph/node metadata). Parameter :obj:`pattern`
        specifies what each axis refers to:

        - 't': temporal dimension;
        - 'n': node dimension;
        - 'c'/'f': channels/features dimension.

        For instance, the pattern of a node-level covariate is 't n f', while a
        pairwise metric between nodes has pattern 'n n'.

        Args:
            name (str): the name of the object. You can then access the added
                object as :obj:`dataset.{name}`.
            value (DataArray): the object to be added. Can be a
                :class:`~pandas.DataFrame`, a :class:`~numpy.ndarray` or a
                :class:`~torch.Tensor`.
            pattern (str, optional): the pattern of the object. A pattern
                specifies what each axis refers to:

                - 't': temporal dimension;
                - 'n': node dimension;
                - 'c'/'f': channels/features dimension.

                If :obj:`None`, the pattern is inferred from the shape.
                (default :obj:`None`)
            add_to_input_map (bool): Whether to map the covariate to dataset
                item when calling :obj:`get` methods.
                (default: :obj:`True`)
            synch_mode (SynchMode): How to synchronize the exogenous variable
                inside dataset item, i.e., with the window slice
                (:obj:`SynchMode.WINDOW`) or horizon (:obj:`SynchMode.HORIZON`).
                It applies only for time-varying covariates.
                (default: :obj:`SynchMode.WINDOW`)
            preprocess (bool): If :obj:`True` and the dataset has a scaler with
                same key, then data are scaled when calling :obj:`get` methods.
                (default: :obj:`True`)
            convert_precision (bool): If :obj:`True`, then cast :attr:`value`
                with dataset's precision.
                (default: :obj:`True`)
        """
        # validate name. name cannot be an attribute of self, but allow override
        self._check_name(name)
        value, pattern = self._parse_covariate(
            value, pattern, name=name, convert_precision=convert_precision
        )
        self._covariates[name] = dict(value=value, pattern=pattern)
        if add_to_input_map:
            self.input_map[name] = BatchMapItem(
                name,
                synch_mode,
                preprocess,
                cat_dim=None,
                pattern=pattern,
                shape=value.size(),
            )

    def update_covariate(
        self,
        name: str,
        value: DataArray | None = None,
        pattern: str | None = None,
        add_to_input_map: bool = True,
        synch_mode: SynchMode | None = None,
        preprocess: bool = None,
    ):
        r"""Update a covariate already in the dataset.

        Args:
            name (str): the name of the object. You can then access the added
                object as :obj:`dataset.{name}`.
            value (DataArray, optional): the object to be added. Can be a
                :class:`~pandas.DataFrame`, a :class:`~numpy.ndarray` or a
                :class:`~torch.Tensor`.
            pattern (str, optional): the pattern of the object. A pattern
                specifies what each axis refers to:

                - 't': temporal dimension;
                - 'n': node dimension;
                - 'c'/'f': channels/features dimension.

                If :obj:`None`, the pattern is inferred from the shape.
                (default :obj:`None`)
            add_to_input_map (bool): Whether to map the covariate to dataset
                item when calling :obj:`get` methods.
                (default: :obj:`True`)
            synch_mode (SynchMode): How to synchronize the exogenous variable
                inside dataset item, i.e., with the window slice
                (:obj:`SynchMode.WINDOW`) or horizon (:obj:`SynchMode.HORIZON`).
                It applies only for time-varying covariates.
                (default: :obj:`SynchMode.WINDOW`)
            preprocess (bool): If :obj:`True` and the dataset has a scaler with
                same key, then data are scaled when calling :obj:`get` methods.
                (default: :obj:`True`)
        """
        # validate name. name cannot be an attribute of self, but allow override
        if name not in self._covariates:
            raise RuntimeError(
                f"There is not a covariate named {name} in "
                f"{self.__class__.__name__}"
            )
        # override value (with or without explicit pattern)
        if value is not None:
            value, pattern = self._parse_covariate(value, pattern, name=name)
        # override pattern, rearranging the stored covariate
        elif pattern is not None:
            pattern = check_pattern(pattern, ndim=value.ndim)
            from einops import rearrange

            old_value = self._covariates[name]["value"]
            old_pattern = self._covariates[name]["pattern"]
            value = rearrange(old_value, f"{old_pattern} -> {pattern}")
        # update the covariate
        self._covariates[name] = dict(value=value, pattern=pattern)
        # update input map
        if add_to_input_map:
            if name in self.input_map:
                kwargs = dict(
                    synch_mode=self.input_map[name].synch_mode,
                    preprocess=self.input_map[name].preprocess,
                )
            else:
                kwargs = dict(synch_mode=None, preprocess=True)
            if preprocess is not None:
                kwargs["preprocess"] = preprocess
            if synch_mode is not None:
                kwargs["synch_mode"] = synch_mode
            shape = tuple(self._covariates[name]["value"].size())
            pattern = self._covariates[name]["pattern"]
            self.input_map[name] = BatchMapItem(
                name, **kwargs, cat_dim=None, pattern=pattern, shape=shape
            )

    def remove_covariate(self, name: str):
        r"""Delete covariate from the dataset.

        Args:
            name (str): the name of the covariate to be deleted.
        """
        try:
            # remove covariate
            del self._covariates[name]
            # remove associated scaler
            if name in self.scalers:
                del self.scalers[name]
            # ATTENTION! remove entirely map item with covariate in keys
            for _map in [self.input_map, self.target_map, self.auxiliary_map]:
                for _map_item in _map:
                    if name in _map_item.keys:
                        del _map[_map_item]
        except Exception as e:
            raise e

    def add_exogenous(
        self,
        name: str,
        value: DataArray,
        node_level: bool = True,
        add_to_input_map: bool = True,
        synch_mode: SynchMode = SynchMode.WINDOW,
        preprocess: bool = True,
    ):
        r"""Shortcut method to add a time-varying covariate.

        Exogenous variables are time-varying covariates of the dataset's target.
        They can either be graph-level (i.e., with same temporal
        length as :obj:`target` but with no node dimension) or node-level (i.e.,
        with same temporal and node size as :obj:`target`).

        Args:
            name (str): The name of the exogenous variable. If the name starts
                with :obj:`"global_"`, the variable is assumed to be graph-level
                (overriding parameter :obj:`node_level`), and the
                :obj:`"global_"` prefix is removed from the name.
            value (DataArray): The data sequence. Can be a
                :class:`~pandas.DataFrame`, a :class:`~numpy.ndarray` or a
                :class:`~torch.Tensor`.
            node_level (bool): Whether the input variable is node- or
                graph-level. If a 2-dimensional array is given and node-level is
                :obj:`True`, it is assumed that the input has one channel.
                (default: :obj:`True`)
            add_to_input_map (bool): Whether to map the exogenous variable to
                dataset item when calling :obj:`get` methods.
                (default: :obj:`True`)
            synch_mode (SynchMode): How to synchronize the exogenous variable
                inside dataset item, i.e., with the window slice
                (:obj:`SynchMode.WINDOW`) or horizon (:obj:`SynchMode.HORIZON`).
                (default: :obj:`SynchMode.WINDOW`)
            preprocess (bool): If :obj:`True` and the dataset has a scaler with
                same key, then data are scaled when calling :obj:`get` methods.
                (default: :obj:`True`)

        Returns:
            SpatioTemporalDataset: the dataset with added exogenous.
        """
        if name.startswith("global_"):
            name = name[7:]
            node_level = False
        pattern = "t n f" if node_level else "t f"
        self.add_covariate(
            name,
            value,
            pattern,
            synch_mode=synch_mode,
            add_to_input_map=add_to_input_map,
            preprocess=preprocess,
        )

    # Setters for preprocessing

    def set_trend(self, trend: DataArray | None):
        r"""Set trend of dataset's target data."""
        if trend is not None:
            trend = self._parse_target(trend)
            # check trend length is equal to target's length
            self._check_same_dim(trend.size(0), "n_steps", "mask")
            # check trend nodes/features are broadcastable to
            # target's nodes/features
            self._check_same_dim(
                trend.size(1), "n_nodes", "mask", allow_broadcasting=True
            )
            self._check_same_dim(
                trend.size(-1), "n_channels", "mask", allow_broadcasting=True
            )
            if "target" in self.scalers:
                if self.__target_bias is None:
                    self.__target_bias = self.scalers["target"].bias
                self.scalers["target"].bias = trend
        else:
            if "target" in self.scalers:
                self.scalers["target"].bias = self.__target_bias
                # refresh maps
                self.update_input_map(self.input_map)
                self.update_target_map(self.target_map)
                # FIX: Added auxiliary_map refresh, missing in TSL
                self.update_auxiliary_map(self.auxiliary_map)
                # END_FIX
        self.trend = trend

    def add_scaler(self, key: str, scaler: Scaler | ScalerModule):
        r"""Add a :class:`tsl.data.preprocessing.Scaler` for the object indexed
        by :obj:`key` in the dataset.

        Args:
            key (str): The name of the variable associated to the scaler. It
                must be a temporal variable, i.e., :obj:`data` or an exogenous.
            scaler (Scaler): The :class:`~tsl.data.preprocessing.Scaler`.
        """
        if key not in self.keys:
            raise KeyError(f"{key} not in {self.name}.")
        # copy to ScalerModule
        scaler = ScalerModule(scaler)
        pattern = self.patterns[key]
        self._check_pattern(
            scaler.bias, pattern, name=f"scaler ({key})", allow_broadcasting=True
        )
        self._check_pattern(
            scaler.scale, pattern, name=f"scaler ({key})", allow_broadcasting=True
        )
        if key == "target" and self.trend is not None:
            self.__target_bias = scaler.bias
            scaler.bias = scaler.bias + self.trend
        scaler.pattern = pattern
        self.scalers[key] = scaler
        # cache batch scaler if target tensor is in a multi-key batch item
        for bm in [self.input_map, self.target_map, self.auxiliary_map]:
            for bm_key, bm_item in bm.items():
                if key in bm_item.keys and len(bm_item.keys) > 1:
                    tensor, scaler = self.collate_keys(
                        bm_item.keys, cat_dim=bm_item.cat_dim, return_pattern=False
                    )
                    self._batch_scalers[bm_key] = scaler

    # Dataset trimming ########################################################

    def reduce(
        self, time_index: IndexSlice | None = None, node_index: IndexSlice | None = None
    ):
        """Reduce the dataset in terms of number of steps and nodes. Returns a
        copy of the reduced dataset.

        If dataset has a connectivity, edges ending to or starting from removed
        nodes will be removed as well.

        Args:
            time_index (IndexSlice, optional): index or mask of the nodes to
                keep after reduction.
                (default: :obj:`None`)
            node_index (IndexSlice, optional): index or mask of the nodes to
                keep after reduction.
                (default: :obj:`None`)
        """
        return deepcopy(self).reduce_(time_index, node_index)

    def reduce_(
        self, time_index: IndexSlice | None = None, node_index: IndexSlice | None = None
    ):
        """Reduce the dataset in terms of number of steps and nodes. This is an
        inplace operation.

        Args:
            time_index (IndexSlice, optional): index or mask of the nodes to
                keep after reduction.
                (default: :obj:`None`)
            node_index (IndexSlice, optional): index or mask of the nodes to
                keep after reduction.
                (default: :obj:`None`)
        """
        # use slice to reduce known tensor
        time_slice = self._get_time_index(time_index, layout="slice")
        node_slice = self._get_node_index(node_index, layout="slice")

        # FIX: when index is not a slice it cannot be converted to slice
        #  but this is silent, so we must make the dimensions broadcastable
        #  if both "slices" are indeed indices
        if (
            not isinstance(time_index, slice)
            and not isinstance(time_slice, slice)
            and not isinstance(node_index, slice)
            and not isinstance(node_slice, slice)
        ):
            time_slice = time_slice.unsqueeze(-1)
            node_slice = node_slice.unsqueeze(0)
        # END_FIX

        # use index to reduce using index-fed functions
        time_index = self._get_time_index(time_index, layout="index")
        node_index = self._get_node_index(node_index, layout="index")
        try:
            self.target = self.target[time_slice, node_slice]

            # FIX: do not attempt if time_index is None
            if self.index is not None and time_index is not None:
                # END_FIX
                self.index = self.index[time_index.numpy()]

            if self.mask is not None:
                self.mask = self.mask[time_slice, node_slice]
            if self.trend is not None:
                self.trend = self.trend[time_slice, node_slice]
            for name, attr in self._covariates.items():
                x, scaler = self.get_tensor(
                    name, time_index=time_index, node_index=node_index
                )
                attr["value"] = x
                if scaler is not None:
                    self.scalers[name] = scaler
        except Exception as e:
            raise e
        return self

    @contextmanager
    def change_windowing(self, **kwargs):
        default, indices = dict(), self._indices
        try:
            assert all([k in _WINDOWING_["all"][1:] for k in kwargs])
            for k, v in kwargs.items():
                default[k] = getattr(self, k)
                setattr(self, k, v)
            yield self
        finally:
            for k, v in default.items():
                setattr(self, k, v)
            self._indices = indices

    # Indexing ################################################################

    def get_item_indices(self, item) -> tuple[dict[str, Tensor], Tensor]:
        base_temporal_indices, node_indices = self._get_base_indices(item)
        temporal_indices = dict()
        temporal_indices["horizon"] = base_temporal_indices + self._horizon_range
        if self.window > 0:
            temporal_indices["window"] = base_temporal_indices + self._window_range
        if self.history > 0:
            shift = self._get_history_shift_for_index(index=base_temporal_indices)
            temporal_indices["history"] = (
                base_temporal_indices - shift
            ) + self._history_range
        return temporal_indices, node_indices

    def get_window_indices(self, item) -> tuple[Tensor, Tensor]:
        base_temporal_indices, node_indices = self._get_base_indices(item)
        return base_temporal_indices + self._window_range, node_indices

    def get_horizon_indices(self, item) -> tuple[Tensor, Tensor]:
        base_temporal_indices, node_indices = self._get_base_indices(item)
        return base_temporal_indices + self._horizon_range, node_indices

    def get_history_indices(self, item: int | slice | Tensor) -> tuple[Tensor, Tensor]:
        base_temporal_indices, node_indices = self._get_base_indices(item)
        shift = self._get_history_shift_for_index(index=base_temporal_indices)
        return (base_temporal_indices - shift) + self._history_range, node_indices

    def _get_base_indices(self, item: int | slice | Tensor) -> tuple[Tensor, Tensor]:
        idx = self._indices[item]
        if idx.dim() == 1:  # item is a single index
            idx = idx.unsqueeze(0)
        # use narrow to keep dim
        base_temporal_indices = torch.narrow(idx, dim=-1, start=0, length=1)
        node_indices = torch.narrow(idx, dim=-1, start=1, length=1)
        return base_temporal_indices, node_indices

    def _get_history_shift_for_index(self, index: Tensor) -> Tensor:
        device = index.device
        size = (index.shape[0], 1) if index.dim() > 1 else (1,)
        if self.history_shift_mode == "random":
            if self.history_shift[0] == self.history_shift[1]:
                return torch.full(
                    size=size, fill_value=self.history_shift[0], device=device
                )
            else:
                # space for min shift is granted by construction
                min_shift, max_shift = self.history_shift
                max_shift = torch.minimum(
                    index - self.history,
                    torch.full(size=size, fill_value=max_shift, device=device),
                )
                return (
                    min_shift
                    + torch.floor(
                        (max_shift - min_shift) * torch.rand(size=size, device=device)
                    ).int()
                )
        elif self.history_shift_mode == "fixed":
            return index - self.history - self.history_shift[0]
        elif self.history_shift_mode == "upto":
            return torch.minimum(
                index - self.history,
                torch.full(size=size, fill_value=self.history_shift[1], device=device),
            )
        else:
            raise ValueError(
                f"Unknown history shift mode {self.history_shift_mode}. "
                "Available modes are: 'random', 'fixed', 'upto'."
            )

    # Representation ##########################################################

    def numpy(self):
        return np.asarray(self.target)

    def dataframe(self):
        columns = pd.MultiIndex.from_product(
            [range(self.n_nodes), range(self.n_channels)], names=["nodes", "channels"]
        )
        data = self.numpy().reshape((-1, self.n_nodes * self.n_channels))
        return pd.DataFrame(data=data, index=self.index, columns=columns)

    # Utilities ###############################################################

    def save(self, filename: str) -> None:
        """Save :obj:`SpatioTemporalDataset` to disk.

        Args:
            filename (str): path to filename for storage.
        """
        torch.save(self, filename)

    @classmethod
    def load(cls, filename: str) -> "TemporalDataset":
        """Load instance of :obj:`TemporalDataset` from disk.

        Args:
            filename (str): path of :obj:`TemporalDataset`.
        """
        obj = torch.load(filename)
        if not isinstance(obj, cls):
            raise TypeError(f"Loaded file is not of class {cls}.")
        return obj

    @classmethod
    def from_dataset(
        cls,
        dataset,
        covariate_keys: list[str] = None,
        input_map: Mapping | BatchMap | None = None,
        target_map: Mapping | BatchMap | None = None,
        auxiliary_map: Mapping | BatchMap | None = None,
        scalers: Mapping[str, Scaler] | None = None,
        trend: DataArray | None = None,
        window: int = 12,
        horizon: int = 1,
        delay: int = 0,
        stride: int = 1,
        window_lag: int = 1,
        horizon_lag: int = 1,
        history: int | None = None,
        history_shift: int | tuple[int, int] | None = None,
    ) -> "TemporalDataset":
        """Create a :class:`TemporalDataset` from a
        :class:`~tsl.datasets.prototypes.TabularDataset`.
        """
        covariates = dataset._covariates
        if covariate_keys is not None:
            covariates = {k: v for k, v in covariates if k in covariate_keys}
        return cls(
            target=dataset.target,
            index=dataset.index,
            mask=dataset.mask,
            covariates=covariates,
            name=dataset.name,
            precision=dataset.precision,
            input_map=input_map,
            target_map=target_map,
            auxiliary_map=auxiliary_map,
            scalers=scalers,
            trend=trend,
            window=window,
            horizon=horizon,
            delay=delay,
            stride=stride,
            window_lag=window_lag,
            horizon_lag=horizon_lag,
            history=history,  # ADDED: history parameter
            history_shift=history_shift,  # ADDED: history shift parameter
        )


def take(
    x: Tensor | list[Tensor],
    in_pattern: str,
    out_pattern: str | None = None,
    time_index: Tensor | None = None,
    node_index: Tensor | None = None,
    *,
    t: int = 1,
    n: int = 1,
    try_index_1d: bool = True,
) -> tuple[Tensor | list[Tensor], str]:
    # validate indices
    assert (time_index is None or time_index.ndim <= 2) and (
        node_index is None or node_index.ndim <= 2
    )

    # validate multi-tensor input
    multi_tensor = isinstance(x, list)
    if multi_tensor:
        assert all(tensor.shape == x[0].shape for tensor in x)
        other_x = x[1:] if len(x) > 1 else []
        x = x[0]
    else:
        other_x = []

    # check patterns
    in_dims = check_pattern(in_pattern, split=True)
    expanded_dims = set()
    if out_pattern is not None:
        out_dims = check_pattern(out_pattern, split=True)
        if not set(in_dims).issubset(out_dims):
            raise RuntimeError(f"Shape {in_dims} cannot be expanded to {out_dims}.")
        # prepare tensor for broadcasting if needed
        if in_dims[0] != "t" and out_dims[0] == "t":
            x = x.unsqueeze(0)
            other_x = [tensor.unsqueeze(0) for tensor in other_x]
            in_dims = ["t"] + in_dims  # add time dimension
            expanded_dims.add("t")
        if in_dims[1] != "n" and out_dims[1] == "n":
            x = x.unsqueeze(1)
            other_x = [tensor.unsqueeze(1) for tensor in other_x]
            in_dims.insert(1, "n")  # add node dimension
            expanded_dims.add("n")
        elif in_dims[0] != "n" and out_dims[0] == "n":
            x = x.unsqueeze(0)
            other_x = [tensor.unsqueeze(0) for tensor in other_x]
            in_dims = ["n"] + in_dims  # add node dimension
            expanded_dims.add("n")
    # ignore index if dimension is not present
    if "t" not in in_dims:
        time_index = None
    if "n" not in in_dims:
        node_index = None
    # expansion values for broadcasting
    if out_pattern is not None:
        dim_expansion_map = dict(
            t=t if time_index is None else time_index.shape[-1],
            n=n if node_index is None else node_index.shape[-1],
        )
    else:
        dim_expansion_map = None

    # compute batch position
    batch_pos = None
    if time_index is not None and time_index.ndim == 2:
        batch_pos = 0
    elif node_index is not None and node_index.ndim == 2:
        batch_pos = 1 if in_dims[0] == "t" else 0
    # handled indices configurations
    # [B, T] with [B, N=1] if specific time steps for specific nodes
    # [T] with [N] if taking the same time steps for some nodes
    if time_index is not None and node_index is not None:
        assert time_index.ndim == node_index.ndim
        if time_index.ndim == 1:  # 1D [T] with [N]
            # make broadcasting possible
            time_index = time_index.unsqueeze(1)
            node_index = node_index.unsqueeze(0)
        else:  # 2D [B, T] with [B, N]
            assert node_index.shape[-1] == 1
            time_index = time_index.unsqueeze(2)
            node_index = node_index.unsqueeze(1)

    indices = dict(
        t=time_index,
        n=node_index,
    )

    # prepare advanced indexing and expand values if there is an output pattern
    advanced_index = []
    expand_values = None if out_pattern is None else []
    for pos, dim in enumerate(in_dims):
        if dim in ["t", "n"] and indices[dim] is not None:
            if dim not in expanded_dims and (try_index_1d or x.shape[pos] != 1):
                advanced_index.append(indices[dim])
                if expand_values is not None:
                    expand_values.append(-1)
            else:
                advanced_index.append(slice(None))
                if expand_values is not None:
                    expand_values.append(dim_expansion_map[dim])
        else:
            advanced_index.append(slice(None))
            if expand_values is not None:
                expand_values.append(dim_expansion_map.get(dim, -1))
    # apply advanced indexing
    x = x[tuple(advanced_index)]
    other_x = [tensor[tuple(advanced_index)] for tensor in other_x]
    # move batch dimension to the first position if needed
    if batch_pos is not None:
        if batch_pos != 0:
            permutation = [batch_pos] + [i for i in range(x.ndim) if i != batch_pos]
            x = x.permute(*permutation)
            other_x = [tensor.permute(*permutation) for tensor in other_x]
        if expand_values is not None:
            expand_values = [-1] + expand_values
    # expand dummy dimensions to index size if needed
    if expand_values is not None:
        x = x.expand(*expand_values)
        other_x = [tensor.expand(*expand_values) for tensor in other_x]
    return x if not multi_tensor else [x] + other_x, " ".join(in_dims)
