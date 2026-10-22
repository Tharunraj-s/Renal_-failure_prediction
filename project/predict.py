from utils import (load_data_predict,
                   preprocess_data,
                   predict_labels,
                   biomarker_questions)

# Function to get user input for biomarkers


def get_biomarker_input():
    biomarkers = []
    for biomarker in biomarker_questions.keys():
        answer = input(biomarker_questions[biomarker] )

        # Convert the answer to a float if it is a number
        if answer.isdigit():
            answer = float(answer)
        # Convert the answer to a 1 if it is yes
        elif answer == 'yes':
            answer = 1
        # Convert the answer to a 0 if it is no
        elif answer == 'no':
            answer = 0
        else:
            # Handle invalid input
            raise ValueError('Invalid answer: {}'.format(answer))

        biomarkers.append(answer)
    return biomarkers

# Function to create a CSV file from the user input


def create_csv_from_input(biomarkers):
    import csv

    with open('biomarkers.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(biomarker_questions.keys())
        writer.writerow(biomarkers)


ImgTest = "Tumor-46-_jpg.rf.43b9c1f0eb4feae6a6b659bf8d23f156.jpg"
# Get user input for biomarkers
biomarkers = get_biomarker_input()
# Create a CSV file from the user input
create_csv_from_input(biomarkers)
df = load_data_predict('biomarkers.csv')
df_ = preprocess_data(df,True)
prediction= predict_labels(df_, ImgTest)