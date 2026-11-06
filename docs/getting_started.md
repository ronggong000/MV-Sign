# Downloading and Using MV-Sign Data

This guide is for using the released artifacts. To reconstruct your own videos, see the separate [extraction and curation guide](../pipeline/README.md).

## Choose a Resource

| Goal | Hugging Face dataset ID | Payload |
|---|---|---|
| Continuous ASL motion | `mage0001/How2Sign-MV3D` | One final SMPL-X NPZ per sample |
| Isolated ASL motion | `mage0001/ASLCitizen-3D` | One final SMPL-X NPZ per sample |
| Original pseudo-gloss labels | `mage0001/How2Sign-PseudoGloss` | Saved labels and English/pseudo-gloss tables |
| Frontal/side reconstruction diagnostic | `mage0001/SGNify-MV` | Rendered videos, render metadata, frame correspondence |

Motion archives do not contain raw RGB videos, intermediate per-view estimates, OBJ sequences, model weights, or licensed SMPL-X/MANO assets. The SGNify resource contains rendered videos, not its original GT meshes or a complete Blender scene.

## Download and Verify

Install `huggingface_hub` and NumPy. Download a complete resource with:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="mage0001/How2Sign-MV3D",
    repo_type="dataset",
    local_dir="datasets/How2Sign-MV3D",
)
```

Full motion releases are large. To inspect the release metadata first, add `allow_patterns=["*.md", "*.json", "*.csv", "SHA256SUMS"]` to the call. This does not download the motion archives. Inspect `samples.csv` and `manifest.json` to choose shards, then download the required `data/*.tar` files from the Files tab or via `hf_hub_download`.

After downloading the full resource:

```bash
cd datasets/How2Sign-MV3D
sha256sum -c SHA256SUMS
mkdir -p extracted
for archive in data/*.tar; do tar -xf "$archive" -C extracted; done
```

The integrity command expects all files listed in `SHA256SUMS`; a metadata-only or partial download will correctly report missing files. Archives use uncompressed tar because the NPZ payloads are already compressed. Keep room for both archives and extracted files, or read archive members directly.

## Read a Motion Without Extracting the Archive

```python
from io import BytesIO
from pathlib import Path
import tarfile
import numpy as np

archive = next(Path("datasets/How2Sign-MV3D/data").glob("*.tar"))
with tarfile.open(archive, "r") as tar:
    member = next(m for m in tar if m.isfile() and m.name.endswith(".npz"))
    with tar.extractfile(member) as stream:
        payload = stream.read()

with np.load(BytesIO(payload), allow_pickle=False) as motion:
    print(member.name)
    print("pose:", motion["poses"].shape)
    print("FPS:", float(motion["mocap_framerate"]))
```

## Motion Conventions

The motion dataset cards contain `array_schema.json` with exact dtypes and representative shapes. Main fields are:

| Field | Interpretation |
|---|---|
| `poses` | `(T, 55, 3)` SMPL-X local axis-angle rotations, in radians; not joint positions |
| `betas` | Fitted body-shape parameters, typically 10 values |
| `mocap_framerate` | Stored sample FPS; read it rather than assuming a fixed frame rate |
| `smplx_expr` | Expression parameters |
| `trans`, `cam_trans` | Distinct saved translation/camera-related fields; neither should be assumed to provide a calibrated world trajectory |
| `smpl_kp3d`, `smplx_joint_proj` | Saved tracking joints/projections; not guaranteed to have been recomputed from the final repaired pose |

The How2Sign release also preserves `source_frame_index`, `source_padding_mask`, `synthetic_bridge_mask`, and hand-validity bookkeeping. Non-sign pruning changes the timeline, and synthetic motion-blend frames do not have a one-to-one source RGB correspondence. Check the masks together; do not interpret every stored source index as an unmodified source frame. These extra fields are not promised for the single-view ASLCitizen release.

You do not need a GPU or the estimator checkpoints to load these arrays. To obtain meshes or forward-kinematic joints, obtain SMPL-X separately under its terms and decode the final rotations and shape consistently. Do not use old saved tracking joints as a substitute for forward kinematics of repaired poses.

## Match Pseudo-Gloss Labels to Motions

The annotation resource retains **35,263 original source-sentence labels**, while the regenerated motion release has **34,939 completed sequences**. This difference is expected: the motion release excludes missing, screened, and failed inputs.

Use the two resources' `samples.csv` files and join on `split` and `sample_id`. Keep only the intersection when creating paired motion/label examples. Do not pair rows by sort order, truncate one list to match the other, or mix validation/test data into training. Labels are automatic pseudo-gloss predictions, not human-verified semantic ground truth.

## Use SGNify-MV

The SGNify-MV archive contains full rendered sequences, not central-only clips. `frame_correspondence.tsv` identifies the original GT OBJ filename, original RGB frame label, and central-range membership for each rendered frame.

GT OBJ filename indices divided by two give the original RGB frame labels, **not zero-based video positions**. Central bounds are inclusive. Use the explicit correspondence rather than comparing the first central prediction to the first full-sequence GT mesh. Original GT meshes must be obtained separately from SGNify.

## Reproduce the Pipeline

The [pipeline guide](../pipeline/README.md) describes estimator setup, raw prediction formats, multi-view curation, the separate single-view utility, and production parameters. Estimator inference requires its own GPU environment and model assets; curation of saved predictions can run on CPU. Pretrained text-to-motion generator weights are not currently part of this release.

Read the [resource terms](data_release.md) and upstream dataset/model terms before use or redistribution.
