# Application repairs

- Models resolve relative to utils.py; startup no longer depends on the working directory.
- Biomarker checkpoints load on CPU with weights-only loading; the network enters evaluation mode.
- Decimal input, finite-value checks and column-specific categorical mappings replace isdigit parsing.
- Missing all biomarkers returns HTTP 400 instead of predicting entirely from defaults.
- Uploaded images are decoded and re-encoded, with an 8 MiB request limit.
- Per-request temporary input files and unique output images prevent cross-request overwrites.
- Image failures return HTTP 503; exceptions are never labelled Normal.
- Multiple detected classes are reported rather than only the last box. No detections are described as a model observation rather than a diagnosis.
- Debug mode is disabled. Missing biomarkers that remain are disclosed as using fixed defaults.

Run `python project/app.py` after installing requirements. The two checkpoints must exist in project/. Seven input/Flask route regression tests pass (inference mocked), including failure responses and request isolation. Python syntax checks pass. Full model inference remains unverified in this workspace because PyTorch/Ultralytics are unavailable. Existing weights and historical metrics have not been altered.

Unique result images are retained under project/static/results. For a public deployment, add authenticated/private result delivery and a retention policy before using actual patient inputs.
