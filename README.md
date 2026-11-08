# SD Forge SVE

This is an Extension for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic), which improves the run-to-run variance for distilled models.

This repository is a fork of [Haoming02/sd-forge-sve](https://github.com/Haoming02/sd-forge-sve) with extra warmup prompt behavior for Z-Image-Turbo style workflows.

### Fork Changes

- Added **Warmup Prompt**: an optional prompt used only during the first SVE steps.
- With **Warmup Blend = 1.0**, Forge runs two passes of the selected sampler and scheduler. The first pass starts from the generated noise, uses **Warmup Prompt**, and stops at **Steps** out of **Warmup Schedule Steps**. The second pass continues that latent at the same step index using the main prompt and the total step count selected in Forge, without adding the initial noise again. For the supplied workflow, set Warmup Schedule Steps to 12, Steps to 2, and Forge Steps to 14.
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
