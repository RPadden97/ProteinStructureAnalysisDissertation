import pandas as pd
import requests
import gzip
from io import BytesIO
from tqdm import tqdm

import multiprocessing as mp
from pandarallel import pandarallel

tqdm.pandas()
pandarallel.initialize(progress_bar=True)

def parse_protein_file(file_path, validation_file=False):
    """
    Parse a protein data file into a structured pandas DataFrame.
    """
    with open(file_path, 'r') as file:
        data = file.read()
    
    entries = data.split("\n[ID]")  # Split entries by [ID] marker
    parsed_data = []
    
    for entry in tqdm(entries):
        if not entry.strip():  # Skip empty entries
            continue
        lines = entry.strip().splitlines()

        # Initialize data fields
        protein_id = ""
        primary_sequence = ""
        evolutionary_data = []
        tertiary_data = []
        mask_data = ""
        
        # Extract ID
        if lines[0].startswith("[ID]"):
            protein_id = lines[1].strip()
        else:
            protein_id = lines[0].strip()

        if validation_file:
            protein_id = protein_id.split("#")[1]

        protein_code = protein_id.split("_")[0]
        
        # Parse the lines
        current_section = None
        for line in lines:
            line = line.strip()
            if line.startswith("[ID]"):
                current_section = "ID"
            elif line.startswith("[PRIMARY]"):
                current_section = "PRIMARY"
            elif line.startswith("[EVOLUTIONARY]"):
                current_section = "EVOLUTIONARY"
                evolutionary_data = []  # Reset evolutionary data
            elif line.startswith("[TERTIARY]"):
                current_section = "TERTIARY"
                tertiary_data = []
            elif line.startswith("[MASK]"):
                current_section = "MASK"
                mask_data = ""
            else:
                # Accumulate data based on the current section
                if current_section == "PRIMARY":
                    primary_sequence = line
                elif current_section == "EVOLUTIONARY":
                    evd = line.split()
                    evolutionary_data.append(evd)
                elif current_section == "TERTIARY":
                    td = line.split()
                    tertiary_data.append(td)
                elif current_section == "MASK":
                    mask_data += line + "\n"
        
        # Append parsed entry
        parsed_data.append({
            "ID": protein_id,
            "CODE": protein_code,
            "PRIMARY": primary_sequence,
            "EVOLUTIONARY": evolutionary_data,
            "TERTIARY": tertiary_data,
            "MASK": mask_data.strip()
        })
    
    # Convert to pandas DataFrame
    df = pd.DataFrame(parsed_data)
    return df

def pdb_info_search(row):
    if row['CODE'] == '' or row['CODE'] is None:
        return None
    
    entry_id = row['CODE']
    url = "https://data.rcsb.org/rest/v1/core/entry/" + entry_id

    response = requests.get(url).json()
    if response.get('status') == '404' or response.get('status') == 404:
        return None
    
    classification = response['struct_keywords']['pdbx_keywords'].split(',') or None
    
    return classification

def get_classification(df):
    df['CLASSIFICATION'] = df.parallel_apply(pdb_info_search, axis=1)
    return df

def print_csv(file_path):
    read_df = pd.read_csv(file_path)
    print(read_df)
    return

def prepare_thinned_sets(file_path, full_set):
    df = parse_protein_file(file_path)
    final = pd.merge(df, full_set[['ID','CLASSIFICATION']], how='left', on='ID')
    output_path = file_path + '.csv'
    final.to_csv(output_path, index=False)
    print(final)
    return final

def testing_ID_match(file_path, testing_data):
    # Load the PDB ID data and testing data
    pdb_id_data = pd.read_csv(file_path)  # Replace with the actual filename of the PDB ID data

    # Merge the datasets on the TAR-ID column
    merged_data = pd.merge(testing_data, pdb_id_data[['Tar-ID', 'PDB ID']], how='left', left_on='CODE', right_on='Tar-ID')

    # Drop rows where PDB_ID is NaN (no match found)
    merged_data = merged_data[~merged_data['PDB ID'].isna()]

    # Overwrite the ID column in the testing data with the PDB_ID column from the PDB data
    merged_data['CODE'] = merged_data['PDB ID']

    # Drop the extra columns added during the merge to maintain the original format
    merged_data.drop(columns=['PDB ID', 'Tar-ID'], inplace=True)

    print("Updated testing data with PDB IDs saved as 'updated_testing_data.csv'")

    return merged_data

files = [
    "./casp7/training_30",
    "./casp7/training_50",
    "./casp7/training_70",
    "./casp7/training_90",
    "./casp7/training_95",
]

# TRAINING 100
print('TRAINING 100 ------------------------')
file_path = "./casp7/training_100"
temp_df = parse_protein_file(file_path)
protein_df = get_classification(temp_df)
output_path = "./casp7/training_100.csv"
df = pd.read_csv('./casp7/training_100.csv')
df = df.drop(columns=["PRIMARY"])

# Training 30
print('TRAINING 30 ------------------------')
file_path = "./casp7/training_30_test"
temp_df = parse_protein_file(file_path)
output_path = "./casp7/training_30_test.csv"
protein_df.to_csv(output_path, index=False)

# TRAINING 50
print('TRAINING 50 ------------------------')
file_path = "./casp7/training_50"
temp_df = parse_protein_file(file_path)
output_path = "./casp7/training_50.csv"
protein_df.to_csv(output_path, index=False)

# TRAINING 70
print('TRAINING 70 ------------------------')
file_path = "./casp7/training_70"
temp_df = parse_protein_file(file_path)
output_path = "./casp7/training_70.csv"
protein_df.to_csv(output_path, index=False)

# TRAINING 90
print('TRAINING 90 ------------------------')
file_path = "./casp7/training_90"
temp_df = parse_protein_file(file_path)
output_path = "./casp7/training_90.csv"
protein_df.to_csv(output_path, index=False)

# TRAINING 95
print('TRAINING 95 ------------------------')
file_path = "./casp7/training_95"
temp_df = parse_protein_file(file_path)
output_path = "./casp7/training_95.csv"
protein_df.to_csv(output_path, index=False)

# Matching over classification to thinned sets
df_100 = pd.read_csv("./casp7/training_100.csv")
for file in files:
    prepare_thinned_sets(file, df_100)

# VALIDATION SET
print('VALIDATION ------------------------')
file_path = "./casp7/validation"
temp_df = parse_protein_file(file_path, True)
protein_df = get_classification(temp_df)
output_path = "./casp7/validation.csv"
protein_df.to_csv(output_path, index=False)

# TEST SET
print('TEST ------------------------')
file_path = "./casp7/testing"
temp_df = parse_protein_file(file_path, True)
temp2_df = testing_ID_match('pdb_ids.csv',temp_df)
protein_df = get_classification(temp2_df)
output_path = "./casp7/testing.csv"
protein_df.to_csv(output_path, index=False)
