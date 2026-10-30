from collections import defaultdict
from typing import Literal, Mapping, Type, Any, Callable

import torch
from torch import nn
from torchmetrics import MetricCollection, Metric

from src.engines.temporal_predictor import TemporalPredictor


class MultiTaskTemporalPredictor(TemporalPredictor):
    def __init__(
        self,
        task_names: list[str],
        model: nn.Module | None = None,
        loss_fn: Callable | None = None,
        scale_target: bool = True,
        metrics: Mapping[str, Metric] | None = None,
        *,
        model_class: Type | None = None,
        model_kwargs: Mapping | None = None,
        optim_class: Type | None = None,
        optim_kwargs: Mapping | None = None,
        scheduler_class: Any | None = None,
        scheduler_kwargs: Mapping | None = None,
    ):
        self.task_names = task_names
        super().__init__(
            model=model,
            model_class=model_class,
            model_kwargs=model_kwargs,
            optim_class=optim_class,
            optim_kwargs=optim_kwargs,
            loss_fn=loss_fn,
            scale_target=scale_target,
            metrics=metrics,
            scheduler_class=scheduler_class,
            scheduler_kwargs=scheduler_kwargs,
        )
        self._eval_step_outputs = defaultdict(lambda: defaultdict(list))

    def _create_and_register_metric(self, metrics, prefix: str) -> MetricCollection:
        mc = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix=prefix,
        )
        return mc

    @staticmethod
    def _compute_prefix(
        stage: Literal["train", "val", "test"],
        dl_name: str | None = None,
        kind: Literal["ind", "transd"] | None = None,
    ) -> str:
        prefix = f"{stage}/"
        if dl_name is not None:
            prefix += f"{dl_name}/"
        if kind is not None:
            prefix += f"{kind}uctive/"
        return prefix

    def _set_metrics(self, metrics):
        self.metrics = nn.ModuleDict()
        for stage in ["train", "val", "test"]:
            prefix = self._compute_prefix(stage=stage)
            self.metrics[prefix] = self._create_and_register_metric(
                metrics=metrics,
                prefix=prefix,
            )
            for dl_name in self.task_names:
                prefix = self._compute_prefix(stage=stage, dl_name=dl_name)
                self.metrics[prefix] = self._create_and_register_metric(
                    metrics=metrics,
                    prefix=prefix,
                )
                for kind in ["ind", "transd"]:
                    prefix = self._compute_prefix(
                        stage=stage, dl_name=dl_name, kind=kind
                    )
                    self.metrics[prefix] = self._create_and_register_metric(
                        metrics=metrics,
                        prefix=prefix,
                    )

    def _get_metrics(
        self,
        stage: Literal["train", "val", "test"],
        dl_name: str | None = None,
        kind: Literal["ind", "transd"] | None = None,
    ):
        return self.metrics[
            self._compute_prefix(stage=stage, dl_name=dl_name, kind=kind)
        ]

    def log_loss(
        self,
        name,
        loss,
        kind: Literal["transd", "ind"] | None = None,
        dl_name: str | None = None,
        **kwargs,
    ):
        self.log(
            f"{self._compute_prefix(stage=name, dl_name=dl_name, kind=kind)}loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs,
        )

    def _compute_and_log_metrics(
        self,
        stage: Literal["train", "val", "test"],
        kind: Literal["ind", "transd"] | None,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        valid_mask: torch.Tensor,
        loss: torch.Tensor | None = None,
        batch_size: int | None = None,
        dl_name: str | None = None,
    ):
        self._get_metrics(stage=stage, dl_name=dl_name, kind=kind).update(
            y_hat, y, valid_mask
        )
        self.log_metrics(
            self._get_metrics(stage=stage, dl_name=dl_name, kind=kind),
            batch_size=batch_size,
            add_dataloader_idx=False,
        )
        if loss is not None:
            self.log_loss(
                name=stage,
                loss=loss,
                dl_name=dl_name,
                batch_size=batch_size,
                kind=kind,
                add_dataloader_idx=False,
            )

    def training_step(self, batch, batch_idx):
        total_loss = torch.tensor(0.0, device=self.device)
        dl_count = 0
        for dl_name, dl_batch in batch.items():
            if dl_batch is None:
                continue
            y_hat, y_hat_loss, y, y_loss = self._shared_step(dl_batch)
            # Compute loss
            loss = self._compute_loss(
                y_hat=y_hat_loss,
                y=y_loss,
                valid_mask=dl_batch.mask,
            )
            # Logging
            self._compute_and_log_metrics(
                stage="train",
                kind=None,
                dl_name=dl_name,
                y_hat=y_hat,
                y=y,
                valid_mask=dl_batch.mask,
                loss=loss,
                batch_size=dl_batch.batch_size,
            )
            total_loss += loss
            dl_count += 1
        total_loss = total_loss / dl_count if dl_count > 0 else total_loss
        self.log_loss(
            name="train",
            loss=total_loss,
            dl_name=None,
            batch_size=None,
            kind=None,
            add_dataloader_idx=False,
        )
        return total_loss

    def _eval_step(
        self, batch, batch_idx: int, dataloader_idx: int, stage: Literal["val", "test"]
    ):
        if isinstance(batch, dict):
            # Joint evaluation (e.g. validation via CombinedLoader)
            for dl_name, dl_batch in batch.items():
                if dl_batch is None:
                    continue
                if not isinstance(dl_batch, list):
                    dl_batch = [dl_batch]
                for idx, i_batch in enumerate(dl_batch):
                    if i_batch is None:
                        continue
                    dataloader_kind = self._dataloader2kind[idx]
                    self._eval_single_batch(
                        i_batch, idx, dataloader_kind, stage, dl_name=dl_name
                    )
        else:
            # Separate evaluation (e.g. testing)
            dl_name = None
            dataloader_kind = "transd"

            datamodule = getattr(self.trainer, "datamodule", None)
            if datamodule is not None and hasattr(datamodule, "get_dataloader_mapping"):
                mapping = datamodule.get_dataloader_mapping(stage)
                if dataloader_idx in mapping:
                    dl_name, dataloader_kind = mapping[dataloader_idx]

            if dl_name is None and dataloader_idx is not None:
                # Fallback mapping
                num_kinds = len(self._dataloader2kind)
                task_idx = dataloader_idx // num_kinds
                kind_idx = dataloader_idx % num_kinds
                if task_idx < len(self.task_names):
                    dl_name = self.task_names[task_idx]
                dataloader_kind = self._dataloader2kind.get(kind_idx, "transd")

            self._eval_single_batch(
                batch, dataloader_idx, dataloader_kind, stage, dl_name=dl_name
            )
        return None

    def _eval_single_batch(
        self,
        batch,
        dataloader_idx: int,
        dataloader_kind: Literal["ind", "transd"],
        stage: Literal["val", "test"],
        dl_name: str | None = None,
    ):
        if self.get_gpi_module() is None:
            y_hat, y_hat_loss, y, y_loss = self._shared_step(batch)
        else:
            y_hat, y_hat_loss, y, y_loss, emb = self._shared_step(
                batch, return_embeddings=True
            )
        # Compute loss
        loss = self._compute_loss(
            y_hat=y_hat_loss,
            y=y_loss,
            valid_mask=batch.mask,
        )
        # Logging
        self._compute_and_log_metrics(
            stage=stage,
            kind=dataloader_kind,
            dl_name=dl_name,
            y_hat=y_hat,
            y=y,
            valid_mask=batch.mask,
            loss=loss,
            batch_size=batch.batch_size,
        )
        if stage == "val" and dl_name is not None:
            self._eval_step_outputs[dl_name][dataloader_kind].append(loss.detach())

    def on_validation_epoch_end(self):
        loss = torch.tensor(0.0, device=self.device)
        num_dls = 0
        for dl_name, dl_outputs in self._eval_step_outputs.items():
            n_components = 0
            if "transd" in dl_outputs:
                transd_loss = torch.stack(dl_outputs["transd"]).mean()
                n_components += 1
            else:
                transd_loss = torch.tensor(0.0, device=self.device)
            if "ind" in dl_outputs:
                ind_loss = torch.stack(dl_outputs["ind"]).mean()
                n_components += 1
            else:
                ind_loss = torch.tensor(0.0, device=self.device)
            if n_components > 0:
                loss += (transd_loss + ind_loss) / n_components
                num_dls += 1
        if num_dls > 0:
            loss = loss / num_dls
        self.log_loss(
            name="val",
            loss=loss,
            dl_name=None,
            batch_size=None,
            kind=None,
            add_dataloader_idx=False,
        )
        self._eval_step_outputs.clear()
