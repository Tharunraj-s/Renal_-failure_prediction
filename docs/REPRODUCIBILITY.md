# Historical pre-repair review

This records the initial upload review. See [REPAIRS.md](REPAIRS.md) for fixes now applied, current entry points and remaining validation requirements.

# Reproduction review

The Flask entry point, utility module and fusion notebook were inspected. Inference and training were not executed; checkpoint compatibility is unverified.

## Existing application behaviour
- Start inside project/. Image inference loads ../best.pt; biomarker inference loads trained_model2.pt.
- The biomarker network is KidneyCNN with 24 inputs and two output classes.
- The image and biomarker predictions are returned together.
- The standalone Fusion.ipynb uses notebook imports and differs from the Flask utility interface; it is not a verified application entry point.

## Issues to resolve before deployment
- app.py uses debug=True.
- Requests share user_biomarkers.csv, image filenames and a single result image. Concurrent requests can overwrite each other's artifacts.
- Numeric parsing uses isdigit(), which does not recognise decimal or signed values as numbers.
- Missing biomarkers are replaced with fixed defaults; disclose that behaviour when evaluating.
- The image path uses the last detected class and falls back to Normal after an exception. An exception is not evidence of a normal scan.
- Checkpoints load without an explicit CPU map_location; verify on CPU-only environments.
- Input validation, upload limits and supported image formats need explicit handling.
- Pin a tested dependency environment after a successful clean run.

These observations document the existing code; this organisation pass does not change model logic or historical results. Existing uploaded images/data and their provenance require verification before redistribution.
