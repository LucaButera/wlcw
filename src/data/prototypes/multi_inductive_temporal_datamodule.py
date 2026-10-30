from typing import Literal

from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from tsl.data.datamodule.spatiotemporal_datamodule import (
    StageOptions,
)

from src.data.prototypes.inductive_temporal_datamodule import (
    InductiveTemporalDataModule,
)


class MultiInductiveTemporalDataModule(LightningDataModule):
    def __init__(
        self,
        datamodules: dict[str, InductiveTemporalDataModule],
        combine_mode: str = "max_size",
        test_separately: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.datamodules = datamodules
        self.combine_mode = combine_mode
        self.test_separately = test_separately
        self.__data_args = kwargs

    def __getattr__(self, item):
        if item in self.__data_args:
            return self.__data_args[item]
        elif item == "datamodules":
            return self.datamodules
        else:
            ds = self.__dict__.get("datamodules")
            for k in ds:
                if hasattr(ds[k], item):
                    return getattr(ds[k], item)
            else:
                raise AttributeError(item)

    def setup(self, stage: StageOptions = None):
        for dm in self.datamodules.values():
            dm.setup(stage=stage)

    def get_dataloader(
        self,
        split: Literal["train", "val", "test"] = None,
        shuffle: bool = False,
        batch_size: int | None = None,
    ) -> CombinedLoader | list | None:
        if split == "test" and self.test_separately:
            flat_loaders = []
            mapping = {}
            idx = 0
            for name, dm in self.datamodules.items():
                loaders = dm.get_dataloader(
                    split=split, shuffle=shuffle, batch_size=batch_size
                )
                if isinstance(loaders, list):
                    for j, loader in enumerate(loaders):
                        kind = (
                            dm._dataloader2kind[j]
                            if hasattr(dm, "_dataloader2kind")
                            else ("transd" if j == 0 else "ind")
                        )
                        flat_loaders.append(loader)
                        mapping[idx] = (name, kind)
                        idx += 1
                elif loaders is not None:
                    flat_loaders.append(loaders)
                    mapping[idx] = (name, "transd")
                    idx += 1
            self._test_dataloader_mapping = mapping
            return flat_loaders

        # For train and val, or when not testing separately, use CombinedLoader
        iterables = {
            name: dm.get_dataloader(split=split, shuffle=shuffle, batch_size=batch_size)
            for name, dm in self.datamodules.items()
        }
        if iterables:
            mode = self.combine_mode if split == "train" else "max_size"
            return CombinedLoader(iterables=iterables, mode=mode)
        return None

    def get_dataloader_mapping(self, stage: str) -> dict:
        if stage == "test":
            return getattr(self, "_test_dataloader_mapping", {})
        return {}

    def train_dataloader(
        self, shuffle: bool = True, batch_size: int | None = None
    ) -> CombinedLoader | None:
        return self.get_dataloader("train", shuffle, batch_size)

    def val_dataloader(
        self, shuffle: bool = False, batch_size: int | None = None
    ) -> CombinedLoader | None:
        return self.get_dataloader("val", shuffle, batch_size)

    def test_dataloader(
        self, shuffle: bool = False, batch_size: int | None = None
    ) -> CombinedLoader | list | None:
        return self.get_dataloader("test", shuffle, batch_size)
