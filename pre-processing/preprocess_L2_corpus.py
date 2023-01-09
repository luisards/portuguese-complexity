import csv
import numpy as np
import pandas as pd

# The data was processed in CTAP in 2 batches
df1 = pd.read_csv("../data/NLI_PT_V3_1.csv", encoding= 'unicode_escape')
df2 = pd.read_csv("../data/NLI_PT_V3_2.csv", encoding= 'unicode_escape')

#Concatenate and pivot data
frames = df1, df2
df = pd.concat(frames)
df = df.pivot_table(index='Text_Title', columns='Feature_Name', values='Value', aggfunc='first').reset_index()

# Extract proficiency level according to file title
def proficiency(row):
  if '_A_' in row['Text_Title']:
    return '1'
  if '_B_' in row['Text_Title']:
    return '2'
  if '_C_' in row['Text_Title']:
    return '3'

# Add proficiency as feature in df
df['Proficiency'] = df.apply (lambda row: proficiency(row), axis=1)

#Save newly formatted data
df.to_csv('../data/NLI-PT_all_features.csv', encoding = 'utf-8-sig')
