# Data Release Status

The two repositories below have been created. Data are being packaged and uploaded; a repository link alone does not mean the upload is complete or publicly accessible.

| Resource | Host | Download URL | SHA-256 manifest |
|---|---|---|---|
| How2Sign-MV3D | Hugging Face | https://huggingface.co/datasets/mage0001/How2Sign-MV3D (upload pending verification) | Included with completed upload |
| ASLCitizen-3D | Hugging Face, private backup | https://huggingface.co/datasets/mage0001/ASLCitizen-3D (not publicly downloadable) | Included with completed upload |
| MM-WLAuslan-MV3D | To be confirmed | Pending | Pending |
| SGNify-MV | To be confirmed | Pending | Pending |

## Packaging Requirements

- The three motion resources contain one final SMPL-X sequence per sample, without per-view intermediate predictions.
- SGNify-MV contains rendered frontal and side videos; the rendering code is distributed separately in the code repository.
- Split large resources into independently downloadable archives and provide an archive manifest, sample identifiers, split assignments where applicable, and SHA-256 checksums.
- Document array fields, shapes, units, frame rates, coordinate conventions, and correspondence to upstream sample identifiers.
- Verify dataset-specific redistribution conditions before publishing download links.
- Retain local source data until the uploaded files have been checked against their checksums.

## Access and Use

How2Sign-MV3D will be made public only after all remote archive sizes and hashes match the local verified package. Its source is How2Sign, whose terms require attribution and noncommercial use. ASLCitizen-3D is held privately pending confirmation of redistribution permission under the original Microsoft Research dataset terms. The other resources and processing/training/rendering code are being organized.
