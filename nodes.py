"""Register the native MOSS-Image SCG and LEAP nodes."""

from comfy_api.latest import ComfyExtension


_CONFLICTING_CFG_OPTIONS = (
    "sampler_cfg_function",
    "sampler_pre_cfg_function",
    "sampler_post_cfg_function",
    "sampler_calc_cond_batch_function",
)


def _validate_model(model, *, feature: str) -> None:
    from comfy.ldm.moss_image.model import MOSSImage
    from comfy.model_patcher import ModelPatcher
    from comfy.model_sampling import CONST, ModelSamplingDiscreteFlow

    if not isinstance(model, ModelPatcher):
        raise ValueError(f"{feature} requires a native ComfyUI ModelPatcher.")
    diffusion_model = model.get_model_object("diffusion_model")
    if not isinstance(diffusion_model, MOSSImage) or hasattr(diffusion_model, "_orig_mod"):
        raise ValueError(f"{feature} requires an uncompiled native MOSS-Image model.")
    sampling = model.get_model_object("model_sampling")
    if (
        not isinstance(sampling, ModelSamplingDiscreteFlow)
        or not isinstance(sampling, CONST)
        or getattr(sampling.calculate_denoised, "__func__", None) is not CONST.calculate_denoised
    ):
        raise ValueError(f"{feature} requires MOSS-Image's native discrete flow prediction.")
    if sampling.multiplier != 1.0:
        raise ValueError(
            f"{feature} requires a native MOSS-Image timestep multiplier of 1. "
            "Remove ModelSamplingSD3 or use ModelSamplingAuraFlow in flow mode."
        )
    for key in _CONFLICTING_CFG_OPTIONS:
        if model.model_options.get(key):
            raise ValueError(f"{feature} cannot be combined with an existing {key} hook.")


class MossGuidanceExtension(ComfyExtension):
    async def get_node_list(self):
        from .leap_nodes import MossImageLEAP, MossImageLEAPCalibrate
        from .scg_nodes import MossImageSCGEuler, MossImageSCGProfile

        return [
            MossImageSCGProfile, MossImageSCGEuler,
            MossImageLEAPCalibrate, MossImageLEAP,
        ]
