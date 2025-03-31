import pandas as pd

# Read the two CSV files
crowd_count_df = pd.read_csv('output/analysis/crowd_count_results/crowd_count_summary.txt')
labels_df = pd.read_csv('labels.csv')

import csv

# Read the crowd count text file into a dictionary
crowd_count = {}
with open('output/analysis/crowd_count_results/crowd_count_summary.txt', 'r') as f:
    for line in f:
        # Split the line at ': '
        filename, count = line.strip().split(': ')
        crowd_count[filename] = int(count)

# Read the labels CSV
labels_df = pd.read_csv('labels.csv')

# Create a new list to store the mapping
mapping_data = []

# Iterate through labels and add crowd count
for _, row in labels_df.iterrows():
    filename = row['Image']
    mrcnn_count = crowd_count.get(filename, 0)  # Default to 0 if not found
    mapping_data.append([
        filename, 
        mrcnn_count, 
        row['Target']
    ])

# Write to mapping CSV
with open('mapping.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Image', 'MRCNN', 'Target'])
    writer.writerows(mapping_data)

print("Mapping CSV created successfully!")