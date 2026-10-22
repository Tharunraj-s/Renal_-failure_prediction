from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import pandas as pd
from utils import (load_data_predict,
                   preprocess_data,
                   predict_labels,
                   save_user_input_to_csv,
                   biomarker_questions)

app = Flask(__name__)
import os

@app.route('/')
def index():
    file_path = "static/results/results.jpg"
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
    except Exception as e:
        pass
    return render_template('index.html', biomarker_questions=biomarker_questions)


@app.route('/predict', methods=['POST'])
def predict():
    # Get user input for biomarkers
    biomarkers = []
    for biomarker in biomarker_questions.keys():
        answer = request.form.get(biomarker)
        # Check if the answer is None or empty
        if answer is None or answer.strip() == '':
            answer = None
        else:
            # Convert user input to the correct data type (e.g., float or int)
            if answer.isdigit():
                answer = float(answer)
            elif answer in ['yes', 'poor', 'normal', 'present']:
                answer = 1
            elif answer in ['no', 'good', 'abnormal', 'not present']:
                answer = 0
        biomarkers.append(answer)

    # Create a DataFrame from user input
    biomarker_df = pd.DataFrame([biomarkers], columns=biomarker_questions.keys())

    # Save the user input to a CSV file with specific column order
    save_user_input_to_csv(biomarker_df, 'user_biomarkers.csv')

    # Handle image upload
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        filename = secure_filename(uploaded_file.filename)
        img_path = 'uploads/' + filename
        uploaded_file.save(img_path)
        image_uploaded= True
    else:
        img_path = None
        image_uploaded = False

    df = load_data_predict('user_biomarkers.csv')
    df_, biomarker_info = preprocess_data(df, True)

    # Make predictions using your `predict_labels` function
    prediction_result = predict_labels(df_, img_path)

    return render_template('result.html', prediction=prediction_result,
                           biomarker_info=biomarker_info, image_uploaded=image_uploaded)


if __name__ == '__main__':
    app.run(debug=True)
