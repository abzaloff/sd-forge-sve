# SD Forge SVE

This is an Extension for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic), which improves the run-to-run variance for distilled models.

This repository is a fork of [Haoming02/sd-forge-sve](https://github.com/Haoming02/sd-forge-sve) with extra warmup prompt behavior for Z-Image-Turbo style workflows.

### Fork Changes

- Added **Warmup Prompt**: an optional prompt used only during the first SVE steps.
- With **Warmup Blend = 1.0**, Forge runs two passes of the selected sampler and scheduler. The first pass starts from the generated noise, uses **Warmup Prompt**, and stops after **SVE Steps**. Its schedule length is calculated automatically as Forge Steps minus SVE Steps (never shorter than the warmup itself), so those early steps move further through the noise schedule. The second pass continues the latent at the same step index on the full Forge Steps schedule with the main prompt, without adding the initial noise again. For example, Forge Steps 14 and SVE Steps 2 use a 12-step warmup schedule followed by the 14-step main schedule.
- Both passes keep Forge's negative prompt conditioning when CFG is above 1. At CFG 1, Forge ignores the negative prompt.
- The two-pass path supports Forge samplers that accept an explicit sigma schedule. Other samplers retain the earlier prompt-switch behavior. **Warmup Blend** below 1.0 also retains the single-schedule blend.
- The original random conditioning-noise mode is still available when **Warmup Prompt** is empty.
- SVE parameters are no longer written to image infotext/metadata.

### Example

- Generate `a photo of a woman` using **Z-Image-Turbo** with the same Seed

<table>
  <tr>
    <th>Default</th>
    <th>SVE</th>
  </tr>
  <tr>
    <td><img src="./before.jpg" width="384"></td>
    <td><img src="./after.jpg" width="384"></td>
  </tr>
</table>

<hr>

- **Reference:** https://github.com/ChangeTheConstants/SeedVarianceEnhancer
