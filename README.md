# SD Forge SVE

This is an Extension for [Forge Neo](https://github.com/Haoming02/sd-webui-forge-classic), which improves the run-to-run variance for distilled models.

This repository is a fork of [Haoming02/sd-forge-sve](https://github.com/Haoming02/sd-forge-sve) with extra warmup prompt behavior for Z-Image-Turbo style workflows.

### Fork Changes

- Added **Warmup Prompt**: an optional prompt used only during the first SVE steps.
- When **Warmup Prompt** is set, the early steps use that prompt conditioning directly, similar to running a short first KSampler pass in ComfyUI and then continuing with the main prompt.
- After the configured SVE steps, the extension stops modifying conditioning and normal generation continues from the latent state created by the warmup steps.
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
