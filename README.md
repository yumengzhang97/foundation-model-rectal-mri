# Pelvic MRI Classification Research Code

This repository contains research code for pelvic MRI preprocessing, segmentation-assisted region extraction, radiomics/clinical data preparation, deep-learning classification, evaluation, and explainability workflows.

The code is shared as a cleaned research-code snapshot. It does not include patient images, derived subject-level tables, trained model weights, logs, or notebook outputs. Before making this repository public, add the final paper citation, DOI, license, and any project-specific usage restrictions required by your institution or collaborators.

## Repository Structure

```text
.
|-- notebooks/                  # Workflow notebooks for data preparation, preprocessing, analysis, and evaluation
|-- othercode/
|   |-- classification_model/    # Python scripts and notebooks for classification experiments
|   `-- gradcam/                 # Grad-CAM and explainability scripts
|-- requirements.txt             # Python dependencies used across the workflows
`-- README.md
```

## What Is Included

- DICOM/NIfTI conversion and MRI orientation checks.
- Resampling, center-cropping, rectum isolation, patch extraction, and train/validation split utilities.
- nnU-Net and TotalSegmentator-related segmentation workflow notebooks.
- Clinical-model analysis, ROC/AUC, confidence interval, and calibration notebooks.
- MONAI/PyTorch classification scripts for 3D pelvic MRI patches.
- Foundation-model and embedding/logistic-regression experiment scripts.
- Grad-CAM scripts for model explainability.

## What Is Not Included

This repository intentionally excludes sensitive and large artifacts:

- Patient imaging data and DICOM metadata.
- Subject-level CSV files, labels, clinical tables, and exported patient information.
- Trained checkpoints, TensorBoard logs, prediction outputs, and generated figures.
- Notebook cell outputs that could contain private paths, IDs, metrics, or data previews.

Keep these artifacts outside version control. If you run the notebooks, clear outputs again before committing.

## Installation

Create a clean Python environment and install the listed dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some workflows require extra setup or separately obtained assets:

- `nnunetv2` workflows need a configured nnU-Net environment and trained model weights.
- TotalSegmentator workflows need a working TotalSegmentator installation and compatible runtime.
- Foundation-model workflows use `medicalmultitaskmodeling` and related dependencies; check upstream licenses and obtain any required weights separately.
- Several scripts contain project-specific absolute paths from the original research environment. Replace these with local paths before running.

## Typical Workflow

The notebooks and scripts are experiment snapshots rather than a single turnkey pipeline. A typical use pattern is:

1. Convert source imaging data to NIfTI and verify orientation.
2. Resample and crop images into the required axial or sagittal views.
3. Generate segmentation masks or region-specific patches where needed.
4. Create train/validation splits and label files outside the repository.
5. Train or run inference with the classification scripts in `othercode/classification_model/`.
6. Evaluate predictions with the AUC, ROC, calibration, and clinical-analysis notebooks.
7. Generate explainability outputs with the Grad-CAM scripts when model checkpoints are available.

## Reproducibility Notes

- Review each notebook/script before execution and update path variables, input directories, label filenames, and output directories.
- The repository records the analysis workflow, but reproducibility depends on access to the original data, preprocessing choices, model weights, package versions, and local compute environment.
- Avoid committing generated outputs. Use `.gitignore` rules or an external artifact store for data, checkpoints, logs, and results.

## Privacy Notice

This project may be associated with sensitive medical data. Do not commit patient data, identifiers, DICOM headers, exported clinical tables, notebook outputs, model predictions linked to subject IDs, or local filesystem paths that reveal private infrastructure.

## Citation

TODO: Add the full article title, authors, journal/conference, year, DOI, and URL.

## License

TODO: Add a license before publishing. Confirm that the selected license is compatible with institutional policy and all upstream dependencies.
