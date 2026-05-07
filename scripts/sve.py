import gradio as gr
import torch
from lib_sve import DecayMethod
from lib_sve.xyz_sve import xyz_support

from modules import prompt_parser, scripts
from modules.processing import StableDiffusionProcessingTxt2Img
from modules.script_callbacks import CFGDenoiserParams, on_cfg_denoiser
from modules.ui_components import InputAccordion


class SeedVarianceEnhancer(scripts.Script):
    sorting_priority = 1125
    MAX_STEPS = 50

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
    warmup_cond = None
    warmup_raw_cond = None

    def __init__(self):
        xyz_support(self.XYZ_CACHE)

    def title(self):
        return "Seed Variance Enhancer"

    def show(self, is_img2img):
        return None if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        with InputAccordion(value=False, label=self.title()) as enable:
            gr.HTML("Improve seed-to-seed image variance for distilled models <b>(i.e. CFG = 1.0)</b>")
            with gr.Row():
                warmup_prompt = gr.Textbox(
                    value="",
                    label="Warmup Prompt (optional)",
                    lines=1,
                    info="if set, the first SVE steps use conditioning from this prompt",
                    scale=1,
                )
                warmup_weight = gr.Slider(
                    value=1.0,
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    label="Warmup Weight",
                    info="strength of Warmup Prompt during early SVE steps",
                    scale=1,
                )
            with gr.Row():
                steps = gr.Slider(value=2, minimum=1, maximum=self.MAX_STEPS, step=1, label="Steps", info="the number of early steps affected by SVE")
                percentage = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Percentage", info="used only without Warmup Prompt")
            with gr.Row():
                strength = gr.Slider(value=18, minimum=0, maximum=64, step=1, label="Strength", info="used only without Warmup Prompt")
                clamping = gr.Slider(value=1.0, minimum=0.0, maximum=1.0, step=0.05, label="Clamping", info="used only without Warmup Prompt")
            decay = gr.Dropdown(
                value="No Decay",
                choices=DecayMethod.choices(),
                label="Decay",
                info="used only without Warmup Prompt",
            )

        return [enable, warmup_prompt, warmup_weight, steps, percentage, strength, decay, clamping]

    def before_process_batch(self, p: StableDiffusionProcessingTxt2Img, enable: bool, warmup_prompt: str, warmup_weight: float, steps: int, percentage: float, strength: int, decay: str, clamping: float, **kwargs):
        enable = bool(self.XYZ_CACHE.get("enable", enable))
        SeedVarianceEnhancer.enable = enable
        if not enable:
            SeedVarianceEnhancer.warmup_prompt = ""
            SeedVarianceEnhancer.warmup_cond = None
            SeedVarianceEnhancer.warmup_raw_cond = None
            self.XYZ_CACHE.clear()
            return

        SeedVarianceEnhancer.warmup_prompt = str(warmup_prompt or "").strip()
        SeedVarianceEnhancer.warmup_weight = float(warmup_weight)
        SeedVarianceEnhancer.warmup_cond = None
        SeedVarianceEnhancer.warmup_raw_cond = None
        SeedVarianceEnhancer.steps = min(int(self.XYZ_CACHE.get("steps", steps)), self.MAX_STEPS)
        SeedVarianceEnhancer.percentage = float(self.XYZ_CACHE.get("percentage", percentage))
        SeedVarianceEnhancer.strength = int(self.XYZ_CACHE.get("strength", strength))
        SeedVarianceEnhancer.decay = str(self.XYZ_CACHE.get("decay", decay))
        SeedVarianceEnhancer.clamping = float(self.XYZ_CACHE.get("clamping", clamping))
        SeedVarianceEnhancer.seed = kwargs["seeds"][0]

        self.XYZ_CACHE.clear()

    def process_batch(self, p: StableDiffusionProcessingTxt2Img, enable: bool, warmup_prompt: str, warmup_weight: float, steps: int, percentage: float, strength: int, decay: str, clamping: float, **kwargs):
        if not SeedVarianceEnhancer.enable or not SeedVarianceEnhancer.warmup_prompt:
            return

        prompts = kwargs.get("prompts") or p.prompts or [SeedVarianceEnhancer.warmup_prompt]
        batch_size = len(prompts)
        warmup_prompts = prompt_parser.SdConditioning(
            [SeedVarianceEnhancer.warmup_prompt] * batch_size,
            width=p.width,
            height=p.height,
            distilled_cfg_scale=p.distilled_cfg_scale,
        )

        try:
            SeedVarianceEnhancer.warmup_raw_cond = p.sd_model.get_learned_conditioning(warmup_prompts)
        except Exception:
            SeedVarianceEnhancer.warmup_raw_cond = None

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
    def same_conditioning_shape(cls, source, target) -> bool:
        if isinstance(source, torch.Tensor) and isinstance(target, torch.Tensor):
            return source.shape == target.shape
        if isinstance(source, dict) and isinstance(target, dict):
            if source.keys() != target.keys():
                return False
            return all(
                not isinstance(target[key], torch.Tensor)
                or (isinstance(source.get(key), torch.Tensor) and source[key].shape == target[key].shape)
                for key in target
            )
        return False

    @classmethod
    def adapt_conditioning(cls, learned, target):
        if isinstance(learned, dict) and isinstance(target, dict):
            adapted = {}
            for key, target_value in target.items():
                source_value = learned.get(key)
                if isinstance(target_value, torch.Tensor):
                    if not isinstance(source_value, torch.Tensor):
                        return None
                    tensor = cls.adapt_tensor_shape(source_value, target_value)
                    if not isinstance(tensor, torch.Tensor):
                        return None
                    adapted[key] = tensor.to(device=target_value.device, dtype=target_value.dtype)
                else:
                    adapted[key] = source_value if source_value is not None else target_value

            return prompt_parser.DictWithShape(adapted) if hasattr(prompt_parser, "DictWithShape") else adapted

        learned_tensor = cls.extract_cond_tensor(learned)
        if not isinstance(learned_tensor, torch.Tensor) or not isinstance(target, torch.Tensor):
            return None

        adapted = cls.adapt_tensor_shape(learned_tensor, target)
        if not isinstance(adapted, torch.Tensor):
            return None

        return adapted.to(device=target.device, dtype=target.dtype)

    @classmethod
    def apply_warmup_weight(cls, warmup, base):
        weight = max(0.0, min(1.0, float(cls.warmup_weight)))
        weight = 1.0 - (1.0 - weight) ** 2
        if weight >= 1.0:
            return warmup
        if weight <= 0.0:
            return base

        if isinstance(warmup, torch.Tensor) and isinstance(base, torch.Tensor):
            return base + (warmup - base) * weight

        if isinstance(warmup, dict) and isinstance(base, dict):
            weighted = {}
            for key, base_value in base.items():
                warmup_value = warmup.get(key)
                if isinstance(base_value, torch.Tensor) and isinstance(warmup_value, torch.Tensor):
                    weighted[key] = base_value + (warmup_value - base_value) * weight
                else:
                    weighted[key] = warmup_value if warmup_value is not None else base_value

            return prompt_parser.DictWithShape(weighted) if hasattr(prompt_parser, "DictWithShape") else weighted

        return warmup

    @classmethod
    @torch.inference_mode()
    def resolve_warmup_cond(cls, params: CFGDenoiserParams, cond: torch.Tensor) -> torch.Tensor:
        if not cls.warmup_prompt:
            return cond

        if cls.warmup_cond is None or not cls.same_conditioning_shape(cls.warmup_cond, cond):
            p: StableDiffusionProcessingTxt2Img = params.denoiser.p
            batch_size = cond.shape[0]
            prompts = prompt_parser.SdConditioning(
                [cls.warmup_prompt] * batch_size,
                width=p.width,
                height=p.height,
                distilled_cfg_scale=p.distilled_cfg_scale,
            )

            try:
                learned = cls.warmup_raw_cond if cls.warmup_raw_cond is not None else p.sd_model.get_learned_conditioning(prompts)
            except Exception:
                cls.warmup_cond = None
                return cond
            cls.warmup_cond = cls.adapt_conditioning(learned, cond)
            if cls.warmup_cond is None:
                return cond

        return cls.apply_warmup_weight(cls.warmup_cond, cond)

    @classmethod
    @torch.inference_mode()
    def on_cfg(cls, params: CFGDenoiserParams):
        if not isinstance(params.denoiser.p, StableDiffusionProcessingTxt2Img) or not cls.enable:
            return
        if params.text_cond is None:
            return
        current_step = getattr(params.denoiser, "step", params.sampling_step)
        total_steps = getattr(params.denoiser, "total_steps", params.total_sampling_steps)
        all_steps: int = min(cls.steps, total_steps)
        if all_steps <= 0:
            return
        if all_steps < current_step:
            return

        cond: torch.Tensor = params.text_cond
        if cls.warmup_prompt:
            params.text_cond = cls.resolve_warmup_cond(params, cond)
            return

        generator = torch.Generator(device=cond.device)
        generator.manual_seed(cls.seed)

        noise_start = torch.clamp(torch.rand(cond.shape, device=cond.device, dtype=cond.dtype, generator=generator), min=-cls.clamping, max=cls.clamping)
        strength = cls.apply_decay(params.sampling_step, all_steps, cls.strength)
        noise = noise_start * 2.0 * strength - strength
        noise_mask = torch.bernoulli(noise_start * cls.percentage, generator=generator).bool()

        modified_noise = noise * noise_mask
        params.text_cond = cond + modified_noise


on_cfg_denoiser(SeedVarianceEnhancer.on_cfg)
