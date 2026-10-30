import numpy as np
import pandas as pd
from tsl.datasets import DatetimeDataset
from tsl.datasets.prototypes.casting import is_datetime_like_index
from tsl.utils import ensure_list


def datetime_encoded(dataset: DatetimeDataset, units: str | list[str]) -> pd.DataFrame:
    r"""Transform dataset's temporal index into covariates using sinusoidal
    transformations. Each temporal unit is used as period to compute the
    operations, obtaining two feature (:math:`\sin` and :math:`\cos`) for
    each unit."""
    assert is_datetime_like_index(dataset.index)
    units = ensure_list(units)
    index_nano = dataset.index.view(np.int64)
    datetime = dict()
    for unit in units:
        nano_unit = time_unit_to_nanoseconds(unit)
        nano_sec = index_nano * (2 * np.pi / nano_unit)
        datetime[unit + "_sin"] = np.sin(nano_sec)
        datetime[unit + "_cos"] = np.cos(nano_sec)
    return pd.DataFrame(datetime, index=dataset.index, dtype=np.float32)


def time_unit_to_nanoseconds(time_unit: str):
    check_time_unit(time_unit)
    if time_unit == "year":
        return 365.2425 * 24 * 60 * 60 * 10**9
    elif time_unit == "month":
        return 30.436875 * 24 * 60 * 60 * 10**9
    elif time_unit == "week":
        time_unit = "W"
    return pd.Timedelta("1" + time_unit).value


def check_time_unit(time_unit: str, include_onehot: bool = False):
    allowed_units = {
        "year",
        "month",
        "week",
        "day",
        "hour",
        "minute",
        "second",
        "millisecond",
        "microsecond",
        "nanosecond",
    }
    if include_onehot:
        allowed_units.update(
            {"month", "weekday", "day_of_week", "dayofweek", "weekofyear"}
        )
    if time_unit not in allowed_units:
        raise RuntimeError(
            f"'{time_unit}' is not a valid time unit. "
            f"Allowed units are {', '.join(allowed_units)}."
        )
