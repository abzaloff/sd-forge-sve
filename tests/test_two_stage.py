"""Check the Forge sampler handoff without loading the Forge UI or a model."""

import ast
import inspect
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "sve.py"
tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
original_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SeedVarianceEnhancer")
methods = [node for node in original_class.body if isinstance(node, ast.FunctionDef) and node.name in {
    "process_before_every_sampling",
}]
test_class = ast.ClassDef(name="SeedVarianceEnhancer", bases=[], keywords=[], body=methods, decorator_list=[])
module = ast.fix_missing_locations(ast.Module(body=[test_class], type_ignores=[]))


class StableDiffusionProcessingTxt2Img:
    pass


class PromptParser:
    ScheduledPromptConditioning = namedtuple("ScheduledPromptConditioning", "end_at_step cond")

    @staticmethod
    def SdConditioning(prompts, **kwargs):
        return prompts

    @staticmethod
    def get_multicond_learned_conditioning(model, prompts, steps):
        # Qwen/Z-Image models return a list of tensors for a batch.
        raw = model.get_learned_conditioning(prompts)
        return SimpleNamespace(batch=[
            [SimpleNamespace(schedules=[PromptParser.ScheduledPromptConditioning(steps, cond)])]
            for cond in raw
        ])


namespace = {
    "inspect": inspect,
    "torch": torch,
    "prompt_parser": PromptParser,
    "StableDiffusionProcessingTxt2Img": StableDiffusionProcessingTxt2Img,
}
exec(compile(module, str(SOURCE), "exec"), namespace)
SVE = namespace["SeedVarianceEnhancer"]


class TwoStageTest(unittest.TestCase):
    def test_handoff_uses_auto_shortened_warmup_schedule_and_main_prompt(self):
        calls = []
        callback_steps = []
        encoded = []
        sampler_started = [False]

        def encode(prompts):
            self.assertFalse(sampler_started[0], "text encoding must precede sampling_prepare")
            encoded.append(tuple(prompts))
            return [torch.ones(2) for _ in prompts]

        def sample(model, x, sigmas, extra_args=None, callback=None, disable=None):
            calls.append((sigmas.clone(), extra_args.copy(), model.steps))
            for index in range(len(sigmas) - 1):
                model.step += 1
                if callback:
                    callback({"i": index})
            return x + 1

        class Sampler:
            func = staticmethod(sample)
            config = SimpleNamespace(total_steps=lambda count: count)

            def get_sigmas(self, p, steps):
                return torch.arange(steps, -1, -1, dtype=torch.float32)

        sampler = Sampler()
        p = StableDiffusionProcessingTxt2Img()
        p.sampler = sampler
        p.steps = 14
        p.prompts = ["user prompt"]
        p.width = p.height = 512
        p.distilled_cfg_scale = 1
        p.is_hr_pass = False
        p.sd_model = SimpleNamespace(
            get_learned_conditioning=encode,
            forge_objects=SimpleNamespace(unet=SimpleNamespace(model_options={"transformer_options": {}})),
        )
        SVE.enable = True
        SVE.warmup_prompt = "abstraction"
        SVE.warmup_weight = 1.0
        SVE.warmup_total_steps = 9  # A stale saved value must have no effect.
        SVE.steps = 2
        SVE().process_before_every_sampling(p)

        self.assertEqual([("abstraction",)], encoded)
        sampler_started[0] = True
        first_schedule = sampler.get_sigmas(p, p.steps)
        model = SimpleNamespace(step=0, steps=14, total_steps=14)
        negative = object()
        output = sampler.func(model, torch.tensor([0.0]), sigmas=first_schedule,
                              extra_args={"cond": "user prompt", "uncond": negative},
                              callback=lambda data: callback_steps.append(data["i"]))

        self.assertEqual([12.0, 11.0, 10.0], calls[0][0].tolist())
        self.assertEqual(list(range(14, -1, -1))[2:], calls[1][0].tolist())
        self.assertTrue(torch.equal(calls[0][1]["cond"].batch[0][0].schedules[0].cond, torch.ones(2)))
        self.assertEqual("user prompt", calls[1][1]["cond"])
        self.assertIs(calls[0][1]["uncond"], negative)
        self.assertIs(calls[1][1]["uncond"], negative)
        self.assertEqual([12, 14], [call[2] for call in calls])
        self.assertEqual(list(range(14)), callback_steps)
        self.assertEqual(2.0, output.item())


if __name__ == "__main__":
    unittest.main()
