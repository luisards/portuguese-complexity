import csv
import numpy as np
import pandas as pd

df1 = pd.read_csv("NLI_PT_V3_1.csv", encoding= 'unicode_escape')
df2 = pd.read_csv("NLI_PT_V3_2.csv", encoding= 'unicode_escape')
frames = df1, df2
df = pd.concat(frames)

df = df.pivot_table(index='Text_Title', columns='Feature_Name', values='Value', aggfunc='first').reset_index()

# Create L1 feature based on text title
def label_l1 (row):
  if row['Text_Title'].startswith('ara'):
    return '1'
  if row['Text_Title'].startswith('chi'):
    return '2'
  if row['Text_Title'].startswith('dut'):
    return '3'
  if row['Text_Title'].startswith('eng'):
    return '4'
  if row['Text_Title'].startswith('fre'):
    return '5'
  if row['Text_Title'].startswith('ger'):
    return '6'
  if row['Text_Title'].startswith('ita'):
    return '7'
  if row['Text_Title'].startswith('jap'):
    return '8'
  if row['Text_Title'].startswith('kor'):
    return '9'
  if row['Text_Title'].startswith('pol'):
    return '10'
  if row['Text_Title'].startswith('rom'):
    return '11'
  if row['Text_Title'].startswith('rus'):
    return '12'
  if row['Text_Title'].startswith('spa'):
    return '13'
  if row['Text_Title'].startswith('swe'):
    return '14'
  if row['Text_Title'].startswith('tet'):
    return '15'

df['L1'] = df.apply (lambda row: label_l1(row), axis=1)

def proficiency(row):
  if '_A_' in row['Text_Title']:
    return '1'
  if '_B_' in row['Text_Title']:
    return '2'
  if '_C_' in row['Text_Title']:
    return '3'

df['Proficiency'] = df.apply (lambda row: proficiency(row), axis=1)

df.to_csv('NLI-PT_all_features.csv', encoding = 'utf-8-sig')

df = df.drop('Lexical Sophistication Feature: Familiarity (FW Token)', axis=1)
df = df.drop('Lexical Sophistication Feature: Familiarity (FW Type)', axis=1)
df = df.drop('Lexical Sophistication Feature: Imageability (FW Token)', axis=1)
df = df.drop('Lexical Sophistication Feature: Imageability (FW Type)', axis=1)
df = df.drop('Lexical Sophistication Feature: Concreteness (FW Type)', axis=1)
df = df.drop('Lexical Sophistication Feature: Age of Acquisition (AW Type)', axis=1)
df = df.drop('Lexical Sophistication Feature: Age of Acquisition (FW Token)', axis=1)
df = df.drop('Lexical Sophistication Feature: Age of Acquisition (FW Type)', axis=1)

df.dropna(inplace=True)

len(df[df.isna().any(axis=1)])
len(df)

df.to_csv('NLI-PT_all_features_noNaN.csv', encoding = 'utf-8-sig')

a = df.columns.tolist()
for i in range(442):
  print(a[i])

for i in range(6):
  print(len(df.loc[df['Number of Syntactic Constituents: Passive Sentences'] == i]))
