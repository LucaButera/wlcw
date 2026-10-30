import math

import torch
from einops import rearrange, repeat, reduce
from torch import Tensor

from src.nn.gpi import GPIModule
from src.nn.models.moirai.module import Moirai2Module
from src.nn.models.temporal.prototypes import GlobalTemporalModel


class MoiraiModel(GlobalTemporalModel):
    def __init__(
        self,
        input_size: int,
        horizon: int,
        window: int,
        history: int = 0,
        output_size: int | None = None,
        exog_size: int = 0,
        point_predictions: bool = True,
        gpi: GPIModule | None = None,
        **kwargs,
    ) -> None:
        assert input_size == 1, "MoiraiModel only supports univariate time series."
        assert exog_size == 0, "MoiraiModel does not support exogenous features."
        assert output_size in (
            None,
            1,
        ), "MoiraiModel only supports univariate time series."
        assert gpi is None, "MoiraiModel does not support GPI."
        super().__init__(
            input_size=input_size,
            horizon=horizon,
            output_size=output_size,
            exog_size=exog_size,
            gpi=None,
            **kwargs,
        )
        self.history = history
        self.window = window
        self.point_predictions = point_predictions
        assert self.emb_size == 0, "MoiraiModel does not support GPI."

        self.module = Moirai2Module.from_pretrained("Salesforce/moirai-2.0-R-small")

    def context_token_length(self, patch_size: int) -> int:
        return math.ceil((self.window + self.history) / patch_size)

    def prediction_token_length(self, patch_size) -> int:
        return math.ceil(self.horizon / patch_size)

    def temporal_processing(
        self,
        x: Tensor,
        u: Tensor | None = None,
        emb: Tensor | None = None,
        h: Tensor | None = None,
    ) -> Tensor:
        # Handle additional inputs
        assert emb is None, "MoiraiModel does not support embeddings."
        if self.history > 0:
            x = torch.cat([h, x], dim=1)

        (
            patched_x,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
        ) = self._preprocess_input(
            self.module.patch_size,
            x,
            past_observed_target=~torch.isnan(x),
            past_is_pad=torch.zeros(x.shape[:2], dtype=torch.bool, device=x.device),
        )

        preds = self.module(
            patched_x,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
        )

        outputs = self._postprocess_output(patched_x, preds)
        return outputs

    def _postprocess_output(
        self,
        patched_x: Tensor,
        preds: Tensor,
    ) -> Tensor:
        per_var_context_token = self.context_token_length(self.module.patch_size)
        total_context_token = self.input_size * per_var_context_token
        per_var_predict_token = self.prediction_token_length(self.module.patch_size)
        total_predict_token = self.input_size * per_var_predict_token

        # In the case of 3 variables, per_var_context_token=3, per_var_predict_token=2.
        # The variate_id is: 0 0 0 1 1 1 2 2 2 [p0] [p0] [p1] [p1] [p2] [p2]
        # pred_index is initialized to be the indices of 2, 5, 8
        # assign_index is initialized to be the indices of 9, 11, 13
        pred_index = torch.arange(
            start=per_var_context_token - 1,
            end=total_context_token,
            step=per_var_context_token,
        )
        assign_index = torch.arange(
            start=total_context_token,
            end=total_context_token + total_predict_token,
            step=per_var_predict_token,
        )
        quantile_prediction = repeat(
            patched_x,
            "... patch_size -> ... num_quantiles patch_size",
            num_quantiles=len(self.module.quantile_levels),
            patch_size=self.module.patch_size,
        ).clone()

        assert (
            per_var_predict_token <= self.module.num_predict_token
        ), "Recursive prediction is not supported"
        preds, adjusted_assign_index = self._structure_multi_predict(
            per_var_predict_token,
            pred_index,
            assign_index,
            preds,
        )
        quantile_prediction[..., adjusted_assign_index, :, :] = preds
        quantile_prediction = self._format_preds(
            self.module.num_quantiles,
            self.module.patch_size,
            quantile_prediction,
            self.input_size,
        )
        if self.point_predictions:
            quantile_prediction = torch.median(quantile_prediction, dim=1).values
        return quantile_prediction

    def _structure_multi_predict(
        self,
        per_var_predict_token,
        pred_index,
        assign_index,
        preds,
    ):
        preds = rearrange(
            preds,
            "... (predict_token num_quantiles patch_size) -> ... predict_token num_quantiles patch_size",
            predict_token=self.module.num_predict_token,
            num_quantiles=self.module.num_quantiles,
            patch_size=self.module.patch_size,
        )
        preds = rearrange(
            preds[..., pred_index, :per_var_predict_token, :, :],
            "... pred_index predict_token num_quantiles patch_size -> ... (pred_index predict_token) num_quantiles patch_size",
        )
        adjusted_assign_index = torch.cat(
            [
                torch.arange(start=idx, end=idx + per_var_predict_token)
                for idx in assign_index
            ]
        )
        return preds, adjusted_assign_index

    @staticmethod
    def _patched_seq_pad(
        patch_size: int,
        x: torch.Tensor,
        dim: int,
        left: bool = True,
        value: float | None = None,
    ) -> torch.Tensor:
        if dim >= 0:
            dim = -x.ndim + dim
        pad_length = -x.size(dim) % patch_size
        if left:
            pad = (pad_length, 0)
        else:
            pad = (0, pad_length)
        pad = (0, 0) * (abs(dim) - 1) + pad
        return torch.nn.functional.pad(x, pad, value=value)

    def _generate_time_id(
        self,
        patch_size: int,
        past_observed_target: torch.Tensor,  # "batch past_seq tgt" | bool
    ) -> tuple[
        torch.Tensor,  # "batch past_token" | int
        torch.Tensor,  # "batch future_token" | int
    ]:
        past_seq_id = reduce(
            self._patched_seq_pad(patch_size, past_observed_target, -2, left=True),
            "... (seq patch) dim -> ... seq",
            "max",
            patch=patch_size,
        )
        past_seq_id = torch.clamp(
            past_seq_id.cummax(dim=-1).values.cumsum(dim=-1) - 1, min=0
        )
        batch_shape = " ".join(map(str, past_observed_target.shape[:-2]))
        future_seq_id = (
            repeat(
                torch.arange(
                    self.prediction_token_length(patch_size),
                    device=past_observed_target.device,
                ),
                f"prediction -> {batch_shape} prediction",
            )
            + past_seq_id.max(dim=-1, keepdim=True).values
            + 1
        )
        return past_seq_id, future_seq_id

    def _preprocess_input(
        self,
        patch_size: int,
        past_target: torch.Tensor,  # "batch past_time tgt" | float
        past_observed_target: torch.Tensor,  # "batch past_time tgt" | bool
        past_is_pad: torch.Tensor,  # "batch past_time" | bool
    ) -> tuple[
        torch.Tensor,  # "batch combine_seq patch" | float # target
        torch.Tensor,  # "batch combine_seq patch" | bool  # observed_mask
        torch.Tensor,  # "batch combine_seq" | int         # sample_id
        torch.Tensor,  # "batch combine_seq" | int         # time_id
        torch.Tensor,  # "batch combine_seq" | int         # variate_id
    ]:
        batch_shape = past_target.shape[:-2]
        device = past_target.device

        target = []
        observed_mask = []
        sample_id = []
        time_id = []
        variate_id = []

        past_seq_id, future_seq_id = self._generate_time_id(
            patch_size, past_observed_target
        )

        future_target = torch.zeros(
            batch_shape
            + (
                self.horizon,
                past_target.shape[-1],
            ),
            dtype=past_target.dtype,
            device=device,
        )
        target.extend(
            [
                rearrange(
                    self._patched_seq_pad(patch_size, past_target, -2, left=True),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                rearrange(
                    self._patched_seq_pad(patch_size, future_target, -2, left=False),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
            ]
        )
        future_observed_target = torch.ones(
            batch_shape
            + (
                self.horizon,
                past_observed_target.shape[-1],
            ),
            dtype=torch.bool,
            device=device,
        )
        observed_mask.extend(
            [
                rearrange(
                    self._patched_seq_pad(
                        patch_size, past_observed_target, -2, left=True
                    ),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
                rearrange(
                    self._patched_seq_pad(
                        patch_size, future_observed_target, -2, left=False
                    ),
                    "... (seq patch) dim -> ... (dim seq) patch",
                    patch=patch_size,
                ),
            ]
        )
        future_is_pad = torch.zeros(
            batch_shape + (self.horizon,),
            dtype=torch.long,
            device=device,
        )
        sample_id.extend(
            [
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, past_is_pad, -1, left=True, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
                repeat(
                    reduce(
                        (
                            self._patched_seq_pad(
                                patch_size, future_is_pad, -1, left=False, value=1
                            )
                            == 0
                        ).int(),
                        "... (seq patch) -> ... seq",
                        "max",
                        patch=patch_size,
                    ),
                    "... seq -> ... (dim seq)",
                    dim=past_target.shape[-1],
                ),
            ]
        )
        time_id.extend(
            [past_seq_id] * past_target.shape[-1]
            + [future_seq_id] * past_target.shape[-1]
        )
        variate_id.extend(
            [
                repeat(
                    torch.arange(past_target.shape[-1], device=device),
                    f"dim -> {' '.join(map(str, batch_shape))} (dim past)",
                    past=self.context_token_length(patch_size),
                ),
                repeat(
                    torch.arange(past_target.shape[-1], device=device),
                    f"dim -> {' '.join(map(str, batch_shape))} (dim future)",
                    future=self.prediction_token_length(patch_size),
                ),
            ]
        )

        target = torch.cat(target, dim=-2)
        observed_mask = torch.cat(observed_mask, dim=-2)
        sample_id = torch.cat(sample_id, dim=-1)
        time_id = torch.cat(time_id, dim=-1)
        variate_id = torch.cat(variate_id, dim=-1)
        return (
            target,
            observed_mask,
            sample_id,
            time_id,
            variate_id,
        )

    def _format_preds(
        self,
        num_quantiles: int,
        patch_size: int,
        preds: torch.Tensor,  # "batch combine_seq patch" | float
        target_dim: int,
    ) -> torch.Tensor:  # "batch num_quantiles future_time *tgt" | float
        start = target_dim * self.context_token_length(patch_size)
        end = start + target_dim * self.prediction_token_length(patch_size)
        preds = preds[..., start:end, :num_quantiles, :patch_size]
        preds = rearrange(
            preds,
            "... (dim seq) num_quantiles patch -> ... num_quantiles (seq patch) dim",
            dim=target_dim,
        )[..., : self.horizon, :]
        return preds
