import gradio as gr
import torch
from lib_sve import DecayMethod
from lib_sve.xyz_sve import xyz_support

from modules import scripts
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
from modules.ui_components import InputAccordion


class SeedVarianceEnhancer(scripts.Script):
    sorting_priority = 1125

    enable: bool = False
    seed: int = -1
    XYZ_CACHE: dict[str, str | float] = {}

    steps: int = -1
    percentage: float = 0.0
    strength: int = 0
    decay: str = None
    clamping: float = 1.0
    warmup_prompt: str = ""
    warmup_weight: float = 1.0
    warmup_cond: torch.Tensor | None = None

    def __init__(self):
        xyz_support(self.XYZ_CACHE)

    def title(self):
        return "Seed Variance Enhancer"

    def show(self, is_img2img):
        return None if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML("Improve seed-to-seed image variance for distilled models <b>(i.e. CFG = 1.0)</b>")
            warmup_prompt = gr.Textbox(
                value="",
                label="Warmup Prompt (optional)",
                lines=1,
                info="if set, the first SVE steps use conditioning from this prompt",
            )
            with gr.Row():
                steps = gr.Slider(value=2, minimum=1, maximum=150, step=1, label="Steps", info="the number of steps to inject random noise")
                percentage = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Percentage", info="the percentage of conditioning to inject random noise")
            with gr.Row():
                strength = gr.Slider(value=18, minimum=0, maximum=64, step=1, label="Strength", info="the strength of the random noise")
                clamping = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Clamping", info="reduce effect strength by clamping the initial noise")
            decay = gr.Dropdown(
                value="No Decay",
                choices=DecayMethod.choices(),
                label="Decay",
                info="apply scaling to the strength based on steps",
            )

        return [enable, warmup_prompt, steps, percentage, strength, decay, clamping]

    def before_process_batch(self, p: StableDiffusionProcessingTxt2Img, enable: bool, warmup_prompt: str, steps: int, percentage: float, strength: int, decay: str, clamping: float, **kwargs):
        enable = bool(self.XYZ_CACHE.get("enable", enable))
        SeedVarianceEnhancer.enable = enable
        if not enable:
            return

        SeedVarianceEnhancer.warmup_prompt = str(warmup_prompt or "").strip()
        SeedVarianceEnhancer.warmup_weight = 1.0
        SeedVarianceEnhancer.warmup_cond = None
        SeedVarianceEnhancer.steps = int(self.XYZ_CACHE.get("steps", steps))
        SeedVarianceEnhancer.percentage = float(self.XYZ_CACHE.get("percentage", percentage))
        SeedVarianceEnhancer.strength = int(self.XYZ_CACHE.get("strength", strength))
        SeedVarianceEnhancer.decay = str(self.XYZ_CACHE.get("decay", decay))
        SeedVarianceEnhancer.clamping = float(self.XYZ_CACHE.get("clamping", clamping))
        SeedVarianceEnhancer.seed = kwargs["seeds"][0]

        self.XYZ_CACHE.clear()

    @classmethod
    def apply_decay(cls, current_step, total_steps, strength):
        function = DecayMethod.decay_function(cls.decay)
        return function(current_step, total_steps, strength)

    @classmethod
    def extract_cond_tensor(cls, learned) -> torch.Tensor | None:
        if isinstance(learned, torch.Tensor):
            return learned
        if isinstance(learned, dict):
            for key in ("crossattn", "c_crossattn", "cond", "conditioning"):
                value = learned.get(key)
                if isinstance(value, torch.Tensor):
                    return value
                if isinstance(value, list) and value and isinstance(value[0], torch.Tensor):
                    return value[0]
            return None
        if isinstance(learned, (tuple, list)):
            for value in learned:
                tensor = cls.extract_cond_tensor(value)
                if isinstance(tensor, torch.Tensor):
                    return tensor
            return None
        return None

    @classmethod
    def adapt_tensor_shape(cls, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        tensor = source

        if tensor.dim() == target.dim() - 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != target.dim():
            return None

        # Match batch dimension
        if tensor.shape[0] != target.shape[0]:
            if tensor.shape[0] == 1:
                tensor = tensor.repeat(target.shape[0], *([1] * (tensor.dim() - 1)))
            elif target.shape[0] == 1:
                tensor = tensor[:1]
            elif target.shape[0] > tensor.shape[0]:
                repeat_count = (target.shape[0] + tensor.shape[0] - 1) // tensor.shape[0]
                tensor = tensor.repeat(repeat_count, *([1] * (tensor.dim() - 1)))[: target.shape[0]]
            else:
                tensor = tensor[: target.shape[0]]

        # Match sequence/channel dims with truncate-or-pad strategy
        for dim in range(1, tensor.dim()):
            src = tensor.shape[dim]
            dst = target.shape[dim]
            if src == dst:
                continue
            if src > dst:
                index = [slice(None)] * tensor.dim()
                index[dim] = slice(0, dst)
                tensor = tensor[tuple(index)]
            else:
                pad_shape = list(tensor.shape)
                pad_shape[dim] = dst - src
                pad = torch.zeros(pad_shape, device=tensor.device, dtype=tensor.dtype)
                tensor = torch.cat((tensor, pad), dim=dim)

        return tensor if tensor.shape == target.shape else None

    @classmethod
    @torch.inference_mode()
    def resolve_warmup_cond(cls, params: CFGDenoiserParams, cond: torch.Tensor) -> torch.Tensor:
        if not cls.warmup_prompt:
            return cond

        if cls.warmup_cond is None or cls.warmup_cond.shape != cond.shape:
            p: StableDiffusionProcessingTxt2Img = params.denoiser.p
            batch_size = cond.shape[0]
            prompts = [cls.warmup_prompt] * batch_size

            try:
                learned = p.sd_model.get_learned_conditioning(prompts)
            except Exception:
                cls.warmup_cond = None
                return cond

            learned_tensor = cls.extract_cond_tensor(learned)
            if not isinstance(learned_tensor, torch.Tensor):
                cls.warmup_cond = None
                return cond

            adapted = cls.adapt_tensor_shape(learned_tensor, cond)
            if not isinstance(adapted, torch.Tensor):
                cls.warmup_cond = None
                return cond

            cls.warmup_cond = adapted.to(device=cond.device, dtype=cond.dtype)

        return cls.warmup_cond

    @classmethod
    @torch.inference_mode()
    def on_cfg(cls, params: CFGDenoiserParams):
        if not isinstance(params.denoiser.p, StableDiffusionProcessingTxt2Img) or not cls.enable:
            return
        if params.text_cond is None:
            return
        all_steps: int = min(cls.steps, params.total_sampling_steps)
        if all_steps <= 0:
            return
        if params.sampling_step >= all_steps:
            return

        cond: torch.Tensor = params.text_cond
        warmup_cond = cls.resolve_warmup_cond(params, cond)
        if cls.warmup_prompt and cls.warmup_weight > 0.0:
            blend_weight = float(max(0.0, min(1.0, cls.warmup_weight)))
            if blend_weight >= 1.0:
                cond = warmup_cond
            else:
                cond = torch.lerp(cond, warmup_cond, blend_weight)
        torch.manual_seed(cls.seed)

        noise_start = torch.clamp(torch.rand_like(cond), min=-cls.clamping, max=cls.clamping)
        strength = cls.apply_decay(params.sampling_step, all_steps, cls.strength)
        noise = noise_start * 2.0 * strength - strength
        noise_mask = torch.bernoulli(noise_start * cls.percentage).bool()

        modified_noise = noise * noise_mask
        params.text_cond = cond + modified_noise


on_cfg_denoiser(SeedVarianceEnhancer.on_cfg)
