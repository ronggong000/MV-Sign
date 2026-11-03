# MV-Sign

**Multi-View 3D Motion Integration and Enhanced Text-to-Motion Conditioning for Sign Language Production**

## Data

| Resource | Contents | Samples | Download |
|---|---|---:|---|
| How2Sign-MV3D | Final SMPL-X motions | 34,939 | [Hugging Face](https://huggingface.co/datasets/mage0001/How2Sign-MV3D) |
| ASLCitizen-3D | Final SMPL-X motions | 83,325 | [Hugging Face](https://huggingface.co/datasets/mage0001/ASLCitizen-3D) |
| How2Sign pseudo-gloss | GPT-5.2 labels, original sentence identifiers | 35,263 source sentences | [Hugging Face](https://huggingface.co/datasets/mage0001/How2Sign-PseudoGloss) |
| SGNify-MV | Rendered frontal/side videos, 30 FPS | 57 signs / 114 videos | [Hugging Face](https://huggingface.co/datasets/mage0001/SGNify-MV) |
| MM-WLAuslan-MV3D | Final SMPL-X motions | 38,580 | In preparation |

How2Sign-MV3D: train **30,903**, validation **1,711**, test **2,325**. Counts describe the regenerated data release, not a rerun of the paper's training experiments. Motion resources contain one final sequence per sample, not intermediate per-view predictions. See each dataset card for archive layout, sample identifiers, frame rates, and checksums. Large files are hosted on Hugging Face rather than committed here.

## Code

- [Motion extraction and curation](pipeline/README.md): AiOS + WiLoR, per-view fusion, cross-view alignment/repair, non-sign pruning, and smoothing.
- Model-training scripts and SGNify rendering code: being organized.

## Use

These are reconstructed motion estimates, not motion-capture ground truth. Upstream raw videos, original meshes, SMPL-X/MANO assets, and pretrained estimator weights are not included. Please cite MV-Sign and the source datasets and follow their respective terms. See [resource details](docs/data_release.md).
