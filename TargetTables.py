import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

# URL of the target webpage
url = "https://www.predictioncenter.org/casp7/targets/cgi/casp7-view.cgi?loc=predictioncenter.org;page=casp7/"

def get_pdb_ids(url):
    # Send GET request to the webpage
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad status codes

    # Parse the HTML content
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract the table rows
    table_rows = soup.find_all('tr')

    # Prepare a list to store extracted data
    data = []

    for row in table_rows:
        columns = row.find_all('td')
        if len(columns) > 6:
            # Extract Tar-ID (first column)
            tar_id = columns[0].get_text(strip=True)

            # Extract the rightmost column's content
            rightmost_column = columns[-1]

            # Check for PDB codes in the column
            pdb_links = rightmost_column.find_all('a', href=True)
            if pdb_links:
                # Extract the first PDB ID from the links
                pdb_id = pdb_links[0].get_text(strip=True)
            else:
                # No PDB ID found
                pdb_id = None

            # Append the data
            data.append({"Tar-ID": tar_id, "PDB ID": pdb_id})

    return data

# Fetch data
data = get_pdb_ids(url)

# Convert to a pandas DataFrame
df = pd.DataFrame(data)

# Debugging: Print first few rows to verify results
print(df.head())

# Save the data to a CSV file
output_csv = "pdb_ids.csv"
df.to_csv(output_csv, index=False)

print(f"Data saved to {output_csv}")