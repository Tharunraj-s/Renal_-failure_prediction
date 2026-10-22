from pathlib import Path
from uuid import uuid4
from tempfile import TemporaryDirectory
import math
from flask import Flask, render_template, request, abort
from PIL import Image, UnidentifiedImageError
import pandas as pd
from utils import load_data_predict, preprocess_data, predict_labels, save_user_input_to_csv, biomarker_questions, column_mappings

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(ROOT / 'templates'), static_folder=str(ROOT / 'static'))
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024
RESULTS = ROOT / 'static' / 'results'


def parse_answer(name, raw):
    value = (raw or '').strip().lower()
    if not value:
        return None
    if name in column_mappings:
        mapping = column_mappings[name]
        if value in mapping:
            return mapping[value]
        if value in ('0', '1'):
            return int(value)
        raise ValueError(f'Invalid value for {name}')
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f'Invalid value for {name}')
    return number


@app.route('/')
def index():
    return render_template('index.html', biomarker_questions=biomarker_questions)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        values = {name: parse_answer(name, request.form.get(name)) for name in biomarker_questions}
    except (ValueError, TypeError) as exc:
        abort(400, description=str(exc))
    if all(value is None for value in values.values()):
        abort(400, description='Enter biomarker values before prediction.')
    uploaded = request.files.get('file')
    has_image = bool(uploaded and uploaded.filename)
    token = uuid4().hex
    result_file = RESULTS / f'{token}.jpg'
    with TemporaryDirectory(prefix='renal-') as folder:
        folder = Path(folder)
        image_path = None
        if has_image:
            image_path = folder / 'scan.png'
            try:
                image = Image.open(uploaded.stream)
                image.verify()
                uploaded.stream.seek(0)
                with Image.open(uploaded.stream) as image:
                    image.convert('RGB').save(image_path)
            except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError):
                abort(400, description='Upload a valid supported image.')
        csv_path = folder / 'biomarkers.csv'
        save_user_input_to_csv(pd.DataFrame([values]), csv_path)
        frame, biomarker_info = preprocess_data(load_data_predict(csv_path), True)
        try:
            prediction = predict_labels(frame, str(image_path) if image_path else None, result_file)
        except Exception:
            app.logger.exception('Model inference failed')
            abort(503, description='Prediction unavailable. Check model files and server logs; no diagnosis was produced.')
    return render_template('result.html', prediction=prediction, biomarker_info=biomarker_info,
                           image_uploaded=has_image, result_image=f'results/{token}.jpg')


if __name__ == '__main__':
    app.run(debug=False)
