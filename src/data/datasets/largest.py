import hashlib

import numpy as np
import pandas as pd
from tsl.datasets import LargeST as _LargeST_


class LargeST(_LargeST_):
    def load(self):
        readings, mask, metadata, adj = super().load()
        # adjust metadata to be numerical
        # district is categorical, convert to string
        if "district" in metadata:
            metadata["district"] = metadata["district"].astype(str)
        # one-hot encode categorical metadata
        metadata = pd.get_dummies(metadata).astype("float32")
        return readings, mask, metadata, adj

    def load_raw(self):
        self.maybe_download()

        filenames = self.required_files_paths

        # load sensors information
        metadata = pd.read_csv(filenames["metadata"], index_col=0)
        max_nodes = len(metadata)

        # possibly select subset, "CA" stands for no subset (whole California)
        node_mask = slice(None)
        # FIX: the key is "district" and not "District" in the metadata
        if self.subset == "GLA":  # Greater Los Angeles
            node_mask = (
                (metadata.district == 7)
                | (metadata.district == 8)
                | (metadata.district == 12)
            ).values
        elif self.subset == "GBA":  # Greater Bay Area
            node_mask = (metadata.district == 4).values
        elif self.subset == "SD":  # San Diego
            node_mask = (metadata.district == 11).values
        metadata = metadata.loc[node_mask]

        # load traffic data only for requested years
        readings = []
        for year in self.years:
            data_path = filenames[str(year)]
            data_df = pd.read_hdf(data_path, key="readings")
            data_df = data_df.loc[:, node_mask]  # filter subset
            # resample here to aggregate only valid observations and
            # align to authors' preprocessing
            if self.freq is not None:
                data_df = data_df.resample(self.freq).mean()
                # in authors' code: data_df.resample('15T').mean().round(0)
            readings.append(data_df)

        readings = (
            readings[0]
            if len(readings) == 1  # avoid useless
            else pd.concat(readings, axis=0)
        )  # computations

        # load adjacency
        edge_index, edge_weight = np.load(filenames["adj"]).values()
        # build square adj from coo to add adj as covariate
        adj = np.eye(max_nodes, dtype=np.float32)
        adj[tuple(edge_index)] = edge_weight
        adj = adj[node_mask][:, node_mask]

        return readings, metadata, adj

    def get_hash(self) -> str:
        s = str(
            (
                self.subset,
                self.years,
                self.freq,
                self.imputation_mode,
            )
        )
        return hashlib.sha256(s.encode("utf-8")).hexdigest()
