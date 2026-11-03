# Resources

Motion packages contain one final SMPL-X NPZ per sample, split into tar archives. `samples.csv`, `manifest.json`, and `SHA256SUMS` describe identifiers, splits, frame counts, and integrity checks. Read per-sample FPS from the saved metadata; do not assume 30 FPS for all datasets.

- [How2Sign-MV3D](https://huggingface.co/datasets/mage0001/How2Sign-MV3D): 34,939 final multi-view-curated sequences, stored in the frontal body coordinate representation.
- [ASLCitizen-3D](https://huggingface.co/datasets/mage0001/ASLCitizen-3D): 83,325 final single-view reconstructions.
- [How2Sign GPT-5.2 pseudo-gloss annotations](https://huggingface.co/datasets/mage0001/How2Sign-PseudoGloss): 35,263 saved source-sentence labels.
- [SGNify-MV rendered videos](https://huggingface.co/datasets/mage0001/SGNify-MV): 57 signs, 114 full-sequence 30-FPS videos with camera/render metadata and explicit frame correspondence.
- MM-WLAuslan-MV3D and model-training/rendering scripts: being organized.

NPZ parameters are predictions, not calibrated motion-capture ground truth. Non-sign removal and motion blending change the correspondence to original RGB frame indices. Consult the dataset card and temporal bookkeeping fields when aligning frames.

## Attribution

Original resources: [How2Sign](https://how2sign.github.io/), [ASL Citizen](https://www.microsoft.com/en-us/research/project/asl-citizen/), and [SGNify](https://sgnify.is.tue.mpg.de/). Their respective dataset and model terms continue to apply; this repository does not relicense upstream materials. How2Sign is published under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/). ASL Citizen terms are available [here](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/). SMPL-X/MANO models and estimator checkpoints must be obtained separately.
