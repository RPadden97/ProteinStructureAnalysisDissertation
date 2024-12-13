import pandas as pd
from tqdm import tqdm

def attach_labels(file_path, filename):
    # Load the datasets
    classification_keywords = pd.read_csv('classification_keywords.csv')
    training_data = pd.read_csv(file_path)

    training_data = training_data.drop(['LABEL'], axis=1, errors='ignore')   

    # Ensure CLASSIFICATION column values are formatted consistently
    classification_keywords['CLASSIFICATION'] = classification_keywords['CLASSIFICATION'].str.strip("[]'").str.strip()
    training_data['CLASSIFICATION'] = training_data['CLASSIFICATION'].str.strip("[]'").str.strip()

    # Merge the datasets on the CLASSIFICATION column
    merged_data = training_data.merge(classification_keywords[['CLASSIFICATION', 'CLUSTER']],
                                    on='CLASSIFICATION', how='left')

    # Rename the CLUSTER column to LABEL
    merged_data.rename(columns={'CLUSTER': 'LABEL'}, inplace=True)

    # Save the updated dataset to a new CSV file
    merged_data.to_csv('./casp7/{}.csv'.format(filename), index=False)

    print(merged_data)
    print("Updated training data with LABEL column saved as '{}.csv'".format(filename))

file_paths = [
    ['./casp7/training_30.csv', 'training_30'],
    ['./casp7/training_50.csv', 'training_50'],
    ['./casp7/training_70.csv', 'training_70'],
    ['./casp7/training_90.csv', 'training_90'],
    ['./casp7/training_95.csv', 'training_95'],
    ['./casp7/training_100.csv', 'training_100'],
    ['./casp7/testing.csv', 'testing'],
    ['./casp7/validation.csv', 'validation'],

]

for file_path in tqdm(file_paths):
    attach_labels(file_path[0], file_path[1])