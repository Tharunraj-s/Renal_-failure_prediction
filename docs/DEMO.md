# Demonstration guide

## Walkthrough
1. Introduce biomarker modelling and kidney-image analysis.
2. Open the analysis PDF to show the original work.
3. Compare classical models using RESULTS.md and the original plots.
4. Show the CNN confusion matrix and loss curve.
5. Open project/templates/index.html and project/utils.py to explain the Flask integration.
6. Explain that both outputs are presented together, without claiming a validated joint fusion model.

## Local application
```bash
python -m pip install -r requirements.txt
python project/app.py
```

Enter biomarker values and optionally upload a supported research example image. The app checkpoints belong in project/. Missing biomarkers use fixed defaults, which the result page discloses.

A report/code walkthrough can be shown before a full environment-backed model run. Original experiment folders, checkpoint variants and evidence are preserved. The original ZIP is in archive/project-original.zip.
