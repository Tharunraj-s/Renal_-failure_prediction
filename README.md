# Maintained implementation

See [repairs and current run instructions](docs/REPAIRS.md) for the corrected entry points and validation limits. Historical notebook outputs are retained as experiment records; they are not results of the corrected implementation.

# Renal Failure Prediction

Academic prototype combining a 24-feature biomarker CNN with YOLO-based kidney-image analysis. The upload includes classical ML comparisons, PyTorch experiments, model checkpoints, figures and a Flask interface.

## Project map

| Area | Location |
|---|---|
| Flask application | [project/](project/) |
| Biomarker experiments and data | [Codes/](Codes/) |
| Fusion notebook | [Fusion.ipynb](Fusion.ipynb) |
| Image-model checkpoint | best.pt |
| Biomarker-model checkpoint used by app | project/trained_model2.pt |
| 30/100 epoch experiment evidence | runs_epoch30/ and runs_epoch100/ |
| File inventory and duplicate groups | [docs/INVENTORY.md](docs/INVENTORY.md) |
| Setup and limitations | [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) |

## Local application setup

Use a separate Python environment and install requirements.txt. Versions are inferred, not locked or runtime-verified.

```bash
python -m pip install -r requirements.txt
cd project
python app.py
```

Run from project/ because existing code loads ../best.pt and trained_model2.pt relative to the working directory. The interface uses the templates and static assets in that directory.

This is a research demonstration. Its outputs have not been clinically validated. The current implementation returns biomarker and image outputs together; that alone does not establish a calibrated multimodal fusion model.

## Research material

The original notebooks, checkpoint variants, ZIP archive and exported outputs are preserved. The two ML-log folders contain separate experimental artifacts and must not be merged based only on similar filenames.

Related publication: [Enhanced Deep Learning Models for Early Detection of Renal Failure from CT Imaging](https://doi.org/10.69888/FTSHSL.2024.000251). The uploaded code has not been verified as an exact reproduction of every published result.
