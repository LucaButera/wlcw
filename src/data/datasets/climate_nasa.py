import os
import hashlib

import numpy as np
import pandas as pd
import omegaconf
from tsl import logger
from tsl.datasets.prototypes import DatetimeDataset
from tsl.ops.similarities import gaussian_kernel
from tsl.utils import download_url, extract_zip


class ClimateNASA(DatetimeDataset):
    r"""Climatic variables readings from satellite observations and reanalizes.

    Data have been collected from https://power.larc.nasa.gov/.

    Two temporal resolutions are available:
        'hourly':
            + Time steps:     8760 (3 years)
            + Channels:       7
            + Sampling rate:  hourly
            + Time interval:  2022-01-01 to 2022-12-31
            + Missing values: YES
        'daily':
            + Time steps:     10958 (30 years)
            + Channels:       10
            + Sampling rate:  daily
            + Time interval:  1991-01-01 to 2022-12-31
            + Missing values: YES

    Two spatial resolutions are available:
        'capitals':
            + nodes: 235 (capitals of the world)
            + setting: sparse
        'us':
            + nodes: 1000 (random in continental USA)
            + setting: dense
    """

    url = {
        (
            "capitals",
            "daily",
        ): "https://drive.switch.ch/index.php/s/v4WHQI2ZKsOID6b/download",
        (
            "capitals",
            "hourly",
        ): "https://drive.switch.ch/index.php/s/p9lKIerVYgRY3gb/download",
        ("us", "daily"): "https://drive.switch.ch/index.php/s/bxuW4sqIW6tYnfU/download",
        (
            "us",
            "hourly",
        ): "https://drive.switch.ch/index.php/s/kRBghMpTTh3xO9C/download",
    }

    similarity_options = {"distance"}

    def __init__(
        self,
        time_res,
        space_res,
        root=None,
        imputation_type="forward",
        freq=None,
        normalize=False,
        target_dims: str
        | list[str]
        | int
        | list[int]
        | tuple
        | np.ndarray
        | None = None,
        exogenous_dims: str
        | list[str]
        | int
        | list[int]
        | tuple
        | np.ndarray
        | None = None,
    ):
        self.time_res = time_res
        self.space_res = space_res
        self.root = root
        self.norm = normalize
        self.name = f"{space_res}_nasa_{time_res}"
        self.imputation_type = imputation_type
        # load dataset and missing values mask
        df, dist, mask = self.load(imputation_type=imputation_type)
        super().__init__(
            target=df,
            mask=mask,
            freq=freq,
            similarity_score="distance",
            temporal_aggregation="nearest",
            name=self.name,
        )

        # get mapping from columns names to column index
        self._columns_mapping = {
            col: i
            for i, col in enumerate(df.columns.get_level_values("channels").unique())
        }

        # --- numpy code
        # # convert to numpy and reshape to (time, nodes, channels)
        # self.target = df.values.reshape(
        #     self.target.shape[0], self.n_nodes, self.n_channels
        # )
        # self.mask = mask.values.reshape(
        #     mask.shape[0], self.n_nodes, self.n_channels
        # )

        # --- multicolumn pandas code
        # Rearrange the data between target and covariates
        target_dims, exogenous_dims = self._parse_feature_dims(
            target_dims, exogenous_dims
        )
        if exogenous_dims is not None:
            assert (
                target_dims is not None
            ), "target_dims should be provided if exogenous_dims is not None"
            exogenous_df = self.target.drop(
                [
                    var
                    for var in df.columns
                    if self._columns_mapping[var[1]] not in exogenous_dims
                ],
                axis=1,
            )
            self.add_exogenous("u", exogenous_df)
        if target_dims is not None:
            target_df = self.target.drop(
                [
                    var
                    for var in df.columns
                    if self._columns_mapping[var[1]] not in target_dims
                ],
                axis=1,
            )
            self.set_target(target_df)
            self.set_mask(self.mask[..., target_dims])
        self.add_covariate("dist", dist, pattern="n n")
        self.target_dims = target_dims
        self.exogenous_dims = exogenous_dims

    def _parse_feature_dims(
        self,
        target_dims: str | list[str] | int | list[int] | tuple | np.ndarray | None,
        exogenous_dims: str | list[str] | int | list[int] | tuple | np.ndarray | None,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        def convert_type(
            dims: int | str | list[int] | list[str] | np.ndarray | None,
        ) -> np.ndarray | None:
            if dims is None:
                return dims
            elif isinstance(dims, int):
                dims = np.array([dims])
            elif isinstance(dims, str):
                # convert str to column indices
                dims = np.array([self._columns_mapping[dims]])
            elif isinstance(dims, omegaconf.listconfig.ListConfig):
                if all(isinstance(d, str) for d in dims):
                    # convert str to column indices
                    dims = np.array([self._columns_mapping[d] for d in dims])
                elif all(isinstance(d, int) for d in dims):
                    dims = np.array(dims)
            else:
                if all(isinstance(d, str) for d in dims):
                    # convert str to column indices
                    dims = np.array([self._columns_mapping[d] for d in dims])
                else:
                    dims = np.array(dims)
            # this also sorts the array
            dims = np.unique(dims)
            return dims

        def check_compatibility(dims: np.ndarray | None):
            if isinstance(dims, np.ndarray):
                assert np.all(
                    dims < self.n_channels
                ), f"dims {dims} should all be less than {self.n_channels}"
            elif dims is None:
                return
            else:
                raise RuntimeError("dims should be a list or np.ndarray")

        target_dims = convert_type(target_dims)
        exogenous_dims = convert_type(exogenous_dims)
        check_compatibility(target_dims)
        check_compatibility(exogenous_dims)
        if isinstance(target_dims, np.ndarray) and isinstance(
            exogenous_dims, np.ndarray
        ):
            assert (
                np.intersect1d(target_dims, exogenous_dims).size == 0
            ), "target_dims and exogenous_dims should not overlap"
        elif target_dims is None and exogenous_dims is not None:
            raise RuntimeError(
                "target_dims should be provided if exogenous_dims is not None"
            )
        return target_dims, exogenous_dims

    def get_hash(self) -> str:
        s = str(
            (
                self.time_res,
                self.space_res,
                self.target_dims,
                self.exogenous_dims,
                self.imputation_type,
                self.norm,
            )
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()

    @property
    def raw_file_names(self):
        return [f"climate_{self.name}.h5"]

    @property
    def required_file_names(self):
        return [
            f"climate_{self.name}.h5",
            f"{self.name}_dist.npy",
        ]

    def download(self) -> None:
        path = download_url(self.url[(self.space_res, self.time_res)], self.root)
        extract_zip(path, self.root)
        os.unlink(path)

    def build(self) -> None:
        self.maybe_download()
        # Build distance matrix
        logger.info("Building distance matrix...")
        raw_file_path = os.path.join(self.root, f"climate_{self.name}.h5")
        # get df
        df = pd.read_hdf(raw_file_path, key="df")
        # get coordinates
        coords = df.columns.get_level_values("nodes").unique()
        lats = [float(value.split(",")[0][1:]) for value in coords]
        lons = [float(value.split(",")[1][:-1]) for value in coords]
        coords = pd.DataFrame({"latitude": lats, "longitude": lons})

        from tsl.ops.similarities import geographical_distance

        dist = geographical_distance(coords, to_rad=True).values

        # Save to built directory
        path = os.path.join(self.root, f"{self.name}_dist.npy")
        np.save(path, (dist).round(3))
        # np.save(path, (dist/1e3).round(3))
        # Remove raw data
        self.clean_downloads()

    def load_raw(self):
        self.maybe_build()
        path = os.path.join(self.root, f"climate_{self.name}.h5")
        # load from file
        df = pd.read_hdf(path, key="df")
        path = os.path.join(self.root, f"{self.name}_dist.npy")
        dist = np.load(path)
        return df, dist

    def load(self, imputation_type: str = "forward"):
        df, dist = self.load_raw()
        df = df.astype(float)

        # NaNs are indicated with -999
        assert not pd.isna(df).any().any()
        mask = ~(df == -999)
        df[~mask] = np.nan

        # normalize
        if self.norm:
            df = self.normalize(df)

        # imputation stage
        if imputation_type == "forward":
            df = df.ffill()
            # some Nans blocks might be at the beginning of a series, impute them backwards
            df = df.bfill()
        elif imputation_type == "zeros":
            # impute missing values with zeros
            df[~mask] = 0.0
        elif imputation_type is None:
            # missing values stay NaNs
            pass
        else:
            raise NotImplementedError("Invalid imputation_type")

        return df, dist, mask

    def normalize(self, df_):
        df = df_.copy()
        locations = [loc for loc in df.columns.get_level_values("nodes")]
        channels = list(set([var for var in df.columns.get_level_values("channels")]))
        for var in channels:
            # get all nodes for one channel
            one_channel_df = df.loc[:, (locations, var)]
            # mean and std avoiding missing values
            mean = np.nanmean(one_channel_df.values)
            std = np.nanstd(one_channel_df.values)
            # standardize and reassign to dataframe
            df.loc[:, (locations, var)] = (one_channel_df - mean) / std
        return df

    def compute_similarity(self, method: str, **kwargs):
        if method == "distance":
            finite_dist = self.dist.reshape(-1)
            finite_dist = finite_dist[~np.isinf(finite_dist)]
            sigma = finite_dist.std()
            return gaussian_kernel(self.dist, sigma)
