# Recorded results

## Classical biomarker models
Values below come from Test_Results.csv in Codes/ML logs copy/models/. The separate ML logs/ run is preserved independently.

| Model | Accuracy (%) | ROC AUC (%) | Source |
|---|---:|---:|---|
| Random Forest | 98.75 | 99.81 | [CSV](../Codes/ML%20logs%20copy/models/RandomForestClassifier/Test_Results.csv) |
| Logistic Regression | 97.50 | 99.87 | [CSV](../Codes/ML%20logs%20copy/models/Logistic%20Regression/Test_Results.csv) |
| Decision Tree | 96.25 | 98.81 | [CSV](../Codes/ML%20logs%20copy/models/DecisionTreeClassifier/Test_Results.csv) |
| KNN | 93.75 | 94.68 | [CSV](../Codes/ML%20logs%20copy/models/KNeighborsClassifier/Test_Results.csv) |

Each CSV repeats aggregate scores alongside two confusion-matrix rows. These are one evaluation per model, not two independent runs.

## Biomarker CNN
The [100-epoch report](../runs_epoch100/classification_report_epoch100.txt) evaluates 80 samples: 52 CKD and 28 Not CKD. It rounds overall accuracy to 0.96; class F1 scores are 0.97 and 0.95 respectively. Keep these results separate from the classical model table.

## Image model
YOLO checkpoints, sample images and outputs are included. No standalone numerical image-evaluation claim is added because its provenance has not been verified here.

These are uploaded experiment records, not a new benchmark run. Dataset splits and preprocessing provenance remain future verification work.
