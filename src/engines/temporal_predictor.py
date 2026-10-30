from collections import defaultdict
from typing import Any, Callable, Mapping, Type, Literal

import torch
from torch import nn
from torchmetrics import Metric, MetricCollection
from tsl.engines import Predictor as _Predictor_

from src.nn.gpi import GPIModule
from src.nn.models.temporal.prototypes import TemporalModel, GlobalTemporalModel


class TemporalPredictor(_Predictor_):
    _dataloader2kind: dict[int, Literal["ind", "transd"]] = {
        0: "transd",
        1: "ind",
    }

    def __init__(
        self,
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
        self._eval_step_outputs = defaultdict(list)

    def get_gpi_module(
        self,
    ) -> GPIModule | None:
        if isinstance(self.model, (TemporalModel, GlobalTemporalModel)):
            return self.model.gpi
        else:
            return None

    def _set_metrics(self, metrics):
        self.train_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="train/",
        )
        self.val_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="val/",
        )
        self.test_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="test/",
        )
        self.transd_train_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="train/transductive/",
        )
        self.transd_val_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="val/transductive/",
        )
        self.transd_test_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="test/transductive/",
        )
        self.ind_train_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="train/inductive/",
        )
        self.ind_val_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="val/inductive/",
        )
        self.ind_test_metrics = MetricCollection(
            metrics={k: self._check_metric(m) for k, m in metrics.items()},
            prefix="test/inductive/",
        )

    def log_loss(
        self, name, loss, kind: Literal["transd", "ind"] | None = None, **kwargs
    ):
        self.log(
            f"{name}/loss" if kind is None else f"{name}/{kind}uctive/loss",
            loss.detach(),
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=False,
            **kwargs,
        )

    def _shared_step(self, batch, return_embeddings=False):
        # Compute outputs and rescale
        y = y_loss = batch.y
        y_hat_loss, emb = self.predict_batch(
            batch,
            preprocess=False,
            postprocess=not self.scale_target,
            return_target=False,
        )
        y_hat = y_hat_loss.detach()
        if self.scale_target:
            y_loss = batch.transform["y"].transform(y)
            y_hat = batch.transform["y"].inverse_transform(y_hat)
        return_values = [y_hat, y_hat_loss, y, y_loss]
        if return_embeddings:
            return_values.append(emb)
        return return_values

    def _compute_loss(
        self,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        valid_mask: torch.Tensor,
    ):
        return self.loss_fn(y_hat, y, valid_mask)

    def _compute_and_log_metrics(
        self,
        stage: Literal["train", "val", "test"],
        kind: Literal["ind", "transd"] | None,
        y_hat: torch.Tensor,
        y: torch.Tensor,
        valid_mask: torch.Tensor,
        loss: torch.Tensor | None,
        batch_size: int | None = None,
    ):
        if kind is not None:
            getattr(self, f"{kind}_{stage}_metrics").update(y_hat, y, valid_mask)
            self.log_metrics(
                getattr(self, f"{kind}_{stage}_metrics"),
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            if loss is not None:
                self.log_loss(
                    stage,
                    loss,
                    batch_size=batch_size,
                    kind=kind,
                    add_dataloader_idx=False,
                )
        else:
            getattr(self, f"{stage}_metrics").update(y_hat, y, valid_mask)
            self.log_metrics(
                getattr(self, f"{stage}_metrics"),
                batch_size=batch_size,
                add_dataloader_idx=False,
            )
            if loss is not None:
                self.log_loss(
                    stage, loss, batch_size=batch_size, add_dataloader_idx=False
                )

    def training_step(self, batch, batch_idx):
        y_hat, y_hat_loss, y, y_loss = self._shared_step(batch)

        # Compute loss
        loss = self._compute_loss(
            y_hat=y_hat_loss,
            y=y_loss,
            valid_mask=batch.mask,
        )

        # Logging
        self._compute_and_log_metrics(
            stage="train",
            kind=None,
            y_hat=y_hat,
            y=y,
            valid_mask=batch.mask,
            loss=loss,
            batch_size=batch.batch_size,
        )
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        return self._eval_step(batch, batch_idx, dataloader_idx, stage="val")

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        return self._eval_step(batch, batch_idx, dataloader_idx, stage="test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        """"""

        if self.get_gpi_module() is None:
            y_hat, _, y, _ = self._shared_step(batch)
            emb = None
        else:
            y_hat, _, y, _, emb = self._shared_step(batch, return_embeddings=True)

        output = dict(y=y, y_hat=y_hat)
        if batch.get("mask") is not None:
            output["mask"] = batch.get("mask")
        if emb is not None:
            output["emb"] = emb
        return output

    def _eval_step(
        self, batch, batch_idx: int, dataloader_idx: int, stage: Literal["val", "test"]
    ):
        dataloader_kind = (
            self._dataloader2kind[dataloader_idx]
            if dataloader_idx is not None
            else "transd"
        )

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
            y_hat=y_hat,
            y=y,
            valid_mask=batch.mask,
            loss=loss,
            batch_size=batch.batch_size,
        )
        self._eval_step_outputs[dataloader_kind].append(loss.detach())
        return loss

    def predict_batch(
        self,
        batch,
        preprocess: bool = False,
        postprocess: bool = True,
        return_target: bool = False,
        **forward_kwargs,
    ):
        inputs, targets, mask, transform = self._unpack_batch(batch)
        if preprocess:
            for key, trans in transform.items():
                if key in inputs:
                    inputs[key] = trans.transform(inputs[key])

        if forward_kwargs is None:
            forward_kwargs = dict()
        out = self.forward(**inputs, **forward_kwargs)
        y_hat, emb = out if isinstance(out, tuple) else (out, None)
        # Rescale outputs
        if postprocess:
            trans = transform.get("y")
            if trans is not None:
                y_hat = trans.inverse_transform(y_hat)
        if return_target:
            y = targets.get("y")
            result = [y, y_hat, mask, emb]
        else:
            result = [y_hat, emb]
        return tuple(result) if len(result) > 1 else result[0]

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        n_components = 0
        if "transd" in self._eval_step_outputs:
            transd_loss = torch.stack(self._eval_step_outputs["transd"]).mean()
            n_components += 1
        else:
            transd_loss = torch.tensor(0.0, device=self.device)
        if "ind" in self._eval_step_outputs:
            ind_loss = torch.stack(self._eval_step_outputs["ind"]).mean()
            n_components += 1
        else:
            ind_loss = torch.tensor(0.0, device=self.device)
        if n_components > 0:
            loss = (transd_loss + ind_loss) / n_components
            # Log average transductive/inductive loss for early stopping
            self.log_loss("val", loss, add_dataloader_idx=False)
        self._eval_step_outputs.clear()
