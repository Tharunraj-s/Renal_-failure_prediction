from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO

biomarker_questions = {
    'age': 'Enter your Age  ',
    'blood_pressure': 'Enter your blood pressure eg.80  ',
    'specific_gravity': 'Enter Blood specific gravity eg.1.02  ',
    'albumin': 'Enter your albumin level [0-Normal 1-Minimal 2-Moderate 3-High 4-Extreme 5-Severe]',
    'sugar': 'Enter your sugar level in urine 0-Normal [1-Minimal 2-Moderate 3-High 4-Extreme 5-Severe]',
    'blood_glucose_random': 'Enter your blood glucose random   ',
    'blood_urea': 'Enter your blood urea level  ',
    'serum_creatinine': 'Enter your serum creatinine level in  mg/dL?   ',
    'sodium': 'Enter your sodium level mEq/L    ',
    'potassium': 'Enter your potassium level mEq/L  ',
    'haemoglobin': 'Enter your hemoglobin level mEq/L   ',
    'packed_cell_volume': 'Enter your packed cell volume %  ',
    'white_blood_cell_count': 'Enter your white blood cell count in cells/μL    ',
    'red_blood_cell_count': 'Enter your red blood cell count in million cells/μL    ',
    'appetite': 'How is your Appetite?  ',
    'red_blood_cells': 'Is your Red blood cells Normal ?     ',
    'pus_cell': 'Do you have pus cells?     ',
    'pus_cell_clumps': 'Do you have pus cell clumps?     ',
    'bacteria': 'Do you have bacteria in your urine?  ',
    'hypertension': 'Do you have hypertension?    ',
    'diabetes_mellitus': 'Do you have diabetes mellitus? ',
    'coronary_artery_disease': 'Do you have coronary artery disease?  ',
    'peda_edema': 'Do you have pedal edema?   ',
    'anemia': 'Do you have anemia? '
}
column_mappings = {
    'red_blood_cells': {'abnormal': 0, 'normal': 1},
    'pus_cell': {'abnormal': 0, 'normal': 1},
    'pus_cell_clumps': {'not present': 0, 'present': 1},
    'bacteria': {'not present': 0, 'present': 1},
    'hypertension': {'no': 0, 'yes': 1},
    'diabetes_mellitus': {'no': 0, 'yes': 1},
    'coronary_artery_disease': {'no': 0, 'yes': 1},
    'appetite': {'good': 0, 'poor': 1},
    'peda_edema': {'no': 0, 'yes': 1},
    'anemia': {'no': 0, 'yes': 1},
    'class':{'ckd':0,'not ckd':1}
}


def save_user_input_to_csv(data, filename='uploads/user_biomarkers.csv'):
    ordered_columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells',
                       'pus_cell','pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea',
                       'serum_creatinine', 'sodium','potassium', 'haemoglobin', 'packed_cell_volume',
                       'white_blood_cell_count', 'red_blood_cell_count','hypertension', 'diabetes_mellitus',
                       'coronary_artery_disease', 'appetite', 'peda_edema','anemia']

    data = data[ordered_columns]

    # Save the DataFrame to a CSV file
    data.to_csv(filename, index=False)



def load_data_predict(file_path):
    Data=file_path
    df=pd.read_csv(Data)
    if 'id' in df.columns:
        df.drop('id', axis=1, inplace=True)

    df.columns = ['age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 'red_blood_cells', 'pus_cell',
                  'pus_cell_clumps', 'bacteria', 'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium',
                  'potassium', 'haemoglobin', 'packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count',
                  'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 'peda_edema',
                  'anemia']

    df['diabetes_mellitus'].replace(to_replace = {'\tno':'no','\tyes':'yes',' yes':'yes'},inplace=True)
    df['coronary_artery_disease'] = df['coronary_artery_disease'].replace(to_replace = '\tno', value='no')
    # converting necessary columns to numerical type
    df['packed_cell_volume'] = pd.to_numeric(df['packed_cell_volume'], errors='coerce')
    df['white_blood_cell_count'] = pd.to_numeric(df['white_blood_cell_count'], errors='coerce')
    df['red_blood_cell_count'] = pd.to_numeric(df['red_blood_cell_count'], errors='coerce')
    # Apply custom mappings to the selected columns
    df.replace(column_mappings, inplace=True)
    return df


def preprocess_data(data,for_prediction=False):
    biomarker_info=[]
    # Convert Series to numpy array
    if not for_prediction:
        X = data.drop("class", axis=1)
        y = data["class"]
    else:
        X = data  # Assuming data contains only features for prediction

    # Define default values for each column
    column_defaults = {
        'age': 48,
        'red_blood_cells': 1,  # Example red blood cell presence (1 for normal, 0 for abnormal)
        'pus_cell': 1,  # Example pus cell presence (1 for normal, 0 for abnormal)
        'pus_cell_clumps': 0,  # Example pus cell clumps presence (1 for present, 0 for absent)
        'bacteria': 0,  # Example bacteria presence (1 for present, 0 for absent)
        'hypertension': 0,  # Example hypertension presence (1 for yes, 0 for no)
        'diabetes_mellitus': 0,  # Example diabetes mellitus presence (1 for yes, 0 for no)
        'coronary_artery_disease': 0,  # Example coronary artery disease presence (1 for yes, 0 for no)
        'appetite': 0,  # Example appetite (0 for good, 1 for poor)
        'peda_edema': 0,  # Example pedal edema presence (1 for yes, 0 for no)
        'anemia': 0, # Example anemia presence (1 for yes, 0 for no)
        'blood_pressure': 80.0,
        'specific_gravity': 1.025,
        'blood_glucose_random': 79.0,
        'blood_urea': 22.0,
        'serum_creatinine': 0.8,
        'sodium': 137.0,
        'potassium': 5.0,
        'haemoglobin': 16.8,
        'packed_cell_volume': 51.0,  # normal packed cell volume percentage
        'white_blood_cell_count': 6000.0,  # normal white blood cell count cells/μL
        'red_blood_cell_count': 6.5,  # normal red blood cell count million cells/μL
        'albumin': 0,
        'sugar': 0,
    }

    # Replace null values with custom default values using an if-else condition
    for col_name in column_defaults.keys():
        default_value = column_defaults[col_name]
        X[col_name] = X[col_name].fillna(default_value)
        
    df = data

    # Define a threshold for numeric values that indicate "normal"
    normal_thresholds = {
        "blood_pressure": [70, 120],
        "specific_gravity": [1.005, 1.030],
        "blood_glucose_random": [70, 140],
        "blood_urea": [6, 20],
        "serum_creatinine": [0.6, 1.2],
        "sodium": [135, 145],
        "potassium": [3.5, 5.0],
        "haemoglobin": [12.0, 16.0],
        "packed_cell_volume": [42, 54],
        "white_blood_cell_count": [4500, 11000],
        "red_blood_cell_count": [4.5, 6.2],
        "albumin": [0, 2],
        "sugar": [0, 1],
    }

    # Loop through each row in the DataFrame
    for index, row in df.iterrows():
        biomarker_values = []
        # print(f"Biomarker Values for Row {index + 1}:")
        for biomarker, default_value in column_defaults.items():
            value = row[biomarker]
            if biomarker in ['red_blood_cells', 'pus_cell']:
                # Biomarkers with 1 for normal and 0 for abnormal
                result = "normal" if value == 1 else "abnormal"
            elif biomarker in ['pus_cell_clumps', 'bacteria']:
                # Biomarkers with 1 for present and 0 for not present
                result = "present" if value == 1 else "not present"
            elif biomarker in ['hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 'peda_edema', 'anemia']:
                # Biomarkers with 1 for yes and 0 for no
                result = "yes" if value == 1 else "no"
            elif biomarker in ['appetite']:
                # Biomarkers with 1 for yes and 0 for no
                result = "poor" if value == 1 else "good"
            elif biomarker in normal_thresholds:
                lower_threshold = normal_thresholds[biomarker][0]
                upper_threshold = normal_thresholds[biomarker][1]
                # Check if the numeric value is within the "normal" range
                if value >= lower_threshold and value <= upper_threshold:
                    result = "normal"
                else:
                    result = "abnormal"
                    # Append the numerical value to the result
                value_str = str(value).rjust(6)
                result = f"{value_str} ({result})"
            else:
                result = str(value)  # Numeric biomarkers
            # Format the output with aligned ":"
            # Define the maximum column width for alignment
            # max_col_width = max(len(biomarker) for biomarker in column_defaults.keys())
            # print(f"{biomarker.ljust(max_col_width)}: {result.rjust(6)}")
            biomarker_values.append(f"{biomarker}: {result}")
        biomarker_info.append(biomarker_values)

    if not for_prediction:
        return X, y
    else:
        return X,biomarker_info

    
def predict_labels(df=None, ImgTest=None, result_path=None):
    
    #Model Load
    root = Path(__file__).resolve().parent
    ImgModel = YOLO(str(root / 'best.pt')) if ImgTest is not None else None
    BioMarker_path = root / 'trained_model2.pt'
    Biomodel = KidneyCNN()
    Biomodel.load_state_dict(torch.load(BioMarker_path, map_location='cpu', weights_only=True))

    Biomodel.eval()

    if df is None and ImgTest is None:
        prediction = "Please give the Image and BioMarker data"
        return prediction
   
    if df is not None:
        # Convert DataFrame to PyTorch tensor
        X_new_tensor = torch.tensor(df.to_numpy(), dtype=torch.float32)

        # Perform predictions
        with torch.no_grad():
            outputs = Biomodel(X_new_tensor)
            _, predicted = torch.max(outputs, 1)

        # Map the predicted class indices to class labels
        class_mapping = {0: 'Chronic Kidney Disease', 1: 'Not Chronic Kidney Disease'}
        prediction1 = [class_mapping[idx] for idx in predicted.tolist()]
        #print(prediction1[0])

    if ImgTest is not None:
        result=ImgModel(ImgTest, imgsz=640, conf=0.5)

        names = ImgModel.names
        detected = []
        prediction2 = 'No abnormality detected by this model'
        for r in result:
            for c in r.boxes.cls:
                detected.append(names[int(c)])
        if detected:
            prediction2 = ", ".join(sorted(set(detected)))
        for r in result:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            #im.show()  # show image
            if result_path is not None:
                destination = Path(result_path)
                destination.parent.mkdir(parents=True, exist_ok=True)
                im.save(destination)  # save image
        #print(prediction2)

    if ImgTest is None:
        prediction2 = "Image not uploaded"

    if df is None:
        prediction1="Enter your Details"

    return prediction1, prediction2

class KidneyCNN(nn.Module):
    def __init__(self, input_size=24, num_classes=2):
        super(KidneyCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * (input_size - 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


