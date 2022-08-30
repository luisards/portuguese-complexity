import csv
import numpy as np
import pandas as pd

df1 = pd.read_csv("../data/EF1_features.csv", encoding= 'unicode_escape')
EF1 = df1.pivot_table(index='Text_Title', columns='Feature_Name', values='Value', aggfunc='first').reset_index()
df2 = pd.read_csv("../data/EF2_features.csv", encoding= 'unicode_escape')
EF2 = df2.pivot_table(index='Text_Title', columns='Feature_Name', values='Value', aggfunc='first').reset_index()
df3 = pd.read_csv("../data/EM_features.csv", encoding= 'unicode_escape')
EM = df3.pivot_table(index='Text_Title', columns='Feature_Name', values='Value', aggfunc='first').reset_index()
df4 = pd.read_csv("../data/ES_features.csv", encoding= 'unicode_escape')
ES = df4.pivot_table(index='Text_Title', columns='Feature_Name', values='Value', aggfunc='first').reset_index()

frames = EF1, EF2, EM, ES
df = pd.concat(frames)

# Extract school level according to title of files
def label_level (row):
  if row['Text_Title'].startswith('1_'):
    return '1'
  if row['Text_Title'].startswith('2_'):
    return '2'
  if row['Text_Title'].startswith('3_'):
    return '3'
  if row['Text_Title'].startswith('4_'):
    return '4'

# Add school level as feature in df
df['Level'] = df.apply (lambda row: label_level(row), axis=1)
df.to_csv('../data/School_levels_exploratory.csv', encoding = 'utf-8-sig')
