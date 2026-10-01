# V5 PatchGAN with MSE reconstruction

This is a fresh 100-epoch run of the original V5 U-Net generator **and**
PatchGAN discriminator. It keeps the original V5 adversarial BCE loss and
replaces the original full-image L1 reconstruction term with full-image MSE:

`generator_loss = adversarial_BCE + 10 × full_image_MSE`

The discriminator uses the original real/fake PatchGAN BCE loss. Both networks
use the original V5 Adam settings (learning rate `2e-4`, betas `0.5, 0.999`).
The generator predicts all three YCrCb channels, including luminance. This is
separate from the generator-only `v5_prism_mse100` run.

The data protocol comes from the completed Prism U-Net MSE run: the exact
COCO train/validation split, 12 degradations, 256-pixel crop, seed 2026,
epoch-addressed training crops, and validation sampling. Batch size is **64**
to improve throughput, compared with Prism's batch size 32.

Training and validation full-image and chroma MSE are recorded every epoch in
`history.jsonl`. `epoch_NNN.pth` saves both networks and both optimizers.
`best.pth` is selected using the lowest **validation chroma MSE**; `last.pth`
is the latest checkpoint. The run is independently resumable.

The default output directory is
`/jet/home/brios/checkpoints/v5_patchgan_mse100`.
Submit or resume with:

```bash
bash scripts/bridges2/submit_v5_patchgan_mse.sh
```
