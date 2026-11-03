# MV-Sign

Multi-View 3D Motion Integration and Enhanced Text-to-Motion Conditioning for Sign Language Production.

Official release repository for MV-Sign. Data packaging and uploads are in progress. Pipeline and training code are being organized for release.

## Resources

| Resource | Release contents | Sequences / signs | Status / host |
|---|---|---:|---|
| How2Sign-MV3D | One final reconstructed SMPL-X motion sequence per sample | 34,939 sequences | [Hugging Face](https://huggingface.co/datasets/mage0001/How2Sign-MV3D): packaging/upload in progress; private until verification |
| ASLCitizen-3D | Final reconstructed SMPL-X motion sequences derived from ASLCitizen videos | 83,325 sequences | [Hugging Face](https://huggingface.co/datasets/mage0001/ASLCitizen-3D): private backup upload in progress; public redistribution clearance pending |
| MM-WLAuslan-MV3D | Final reconstructed SMPL-X motion sequences | 38,580 sequences | Pending |
| SGNify-MV | Frontal- and side-view videos rendered from SGNify ground-truth meshes | 57 signs | Pending |

How2Sign-MV3D contains 30,903 training, 1,711 validation, and 2,325 test sequences. These regenerated-release counts do not replace the historical training counts underlying the submitted experiments.

The following materials are being organized: the motion-curation pipeline, model-training scripts, GPT-5.2-generated How2Sign pseudo-glosses, MM-WLAuslan-MV3D, SGNify-MV rendered videos, and SGNify multi-view rendering scripts. No release date is asserted for unfinished items.

## Downloads

Large motion archives and rendered videos are hosted separately, not committed to Git. See [data release status](docs/data_release.md).

## Motion Format

Each motion sample is an NPZ containing SMPL-X pose and shape parameters, plus saved auxiliary fields. The three motion resources are reconstructed predictions, not motion-capture ground truth. The How2Sign release retains one final sequence per sample in the frontal body coordinate representation after cross-view hand checking and repair. Internal side-view intermediate results are not included.

Read the dataset card and `array_schema.json` for exact fields and conventions. Read FPS per sample, rather than assuming 30 FPS. The release archives include `samples.csv`, `manifest.json`, and `SHA256SUMS` for identifiers, split assignments, frame counts, and integrity verification. Do not assume that released frame indices equal original RGB indices after non-sign pruning and motion blending.

## Source Assets and Terms

The release does not bundle upstream raw videos, original ground-truth meshes, or SMPL-X/MANO model files. Source assets and model files remain subject to their original access and license terms. Dataset-specific redistribution conditions will be documented before public release; no blanket dataset license is granted by this draft README.

How2Sign's [source terms](https://how2sign.github.io/) specify CC BY-NC 4.0. Follow attribution and noncommercial-use requirements. ASL Citizen's [dataset terms](https://www.microsoft.com/en-us/research/project/asl-citizen/dataset-license/) restrict redistribution of the data and modifications; the private backup link is not a public download or a claim of redistribution permission. SMPL-X/MANO assets must be obtained separately under their own licenses.

Please cite the source datasets and MV-Sign when using released resources. The final MV-Sign bibliographic entry and code instructions are being prepared.
