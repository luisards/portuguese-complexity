import csv
import pandas as pd
from matplotlib import pyplot as plt


#L1
df1 = pd.read_csv("../data/School_levels_exploratory.csv", encoding= 'unicode_escape')
l1_feature_sample = ['Number of Tokens', 'Number of Sentences', 'Mean Sentence Length in Tokens', 'Mean Token Length in Syllables',
                     'Lexical Richness: Type Token Ratio (Corrected TTR)', 'Lexical Sophistication Feature: Age of Acquisition (AW Token)',
                     'Lexical Sophistication Feature: Concreteness (AW Type)',  'Lexical Variation Feature: Corrected Verb Variation 1',
                     'Morphological Complexity Inflection Feature: Past Participle Verb Density','Number of Morphological Features: Imperfect Verb per VP',
                     'Number of Syntactic Constituents: Passive Sentences', 'Number of POS Feature: Noun Tokens']

for feature in l1_feature_sample:
  a = df1.loc[df1['Level'] == 1, feature]
  b = df1.loc[df1['Level'] == 2, feature]
  c = df1.loc[df1['Level'] == 3, feature]
  d = df1.loc[df1['Level'] == 3, feature]
  columns = [a, b, c, d]
  #plt.rcParams["figure.figsize"] = (7,7)
  fig, ax = plt.subplots()
  #plt.ylim(0, 1300)
  plt.ylabel(feature)
  plt.xlabel('School level')
  ax.boxplot(columns, notch=True, showfliers=True)
  #plt.show()
  feature_name = feature.replace(" ", "_")
  plt.savefig('L1_'+feature_name+'.png')
  plt.close(fig)

#L2
l2_feature_sample = ['Number of Tokens', 'Number of Sentences', 'Mean Sentence Length in Tokens', 'Mean Token Length in Syllables',
                  'Lexical Richness: Type Token Ratio (Corrected TTR)', 'Lexical Sophistication Feature: Age of Acquisition (AW Token)',
                  'Lexical Sophistication Feature: Concreteness (AW Token)', 'Lexical Sophistication Feature: Imageability (AW Token)',
                  'Lexical Sophistication Feature: SUBTLEX Frequency Top 6000 and Below', 'Lexical Variation Feature: Corrected Verb Variation 1',
                  'POS Density Feature: Auxiliary Verb','Morphological Complexity Inflection Feature: Indicatives per Verb', 'Morphological Complexity Inflection Feature: Indicatives per word token',
                  'Morphological Complexity Inflection Feature: Past Participle Verb Density', 'Number of Syntactic Constituents: Passive Sentences', 'Number of Syntactic Constituents: Dependent Clauses', 'Syntactic Complexity Feature: Complex Nominals per Sentence',
                  'Syntactic Complexity Feature: Mean Length of Clause','Syntactic Complexity Feature: Mean Length of Prepositional Phrase',
                  'Number of Connectives: Mendes All Connectives']
df2 = pd.read_csv("../data/NLI-PT_all_features.csv", encoding= 'unicode_escape')

for feature in l2_feature_sample:
  a = df2.loc[df2['Proficiency'] == 1, feature]
  b = df2.loc[df2['Proficiency'] == 2, feature]
  c = df2.loc[df2['Proficiency'] == 3, feature]
  columns = [a, b, c]
  #plt.rcParams["figure.figsize"] = (7,7)
  fig, ax = plt.subplots()
  #plt.ylim(0, 1300)
  plt.ylabel(feature)
  plt.xlabel('Proficiency level')
  ax.boxplot(columns, notch=True, showfliers=True)
#  plt.show()
  feature_name = feature.replace(" ", "_")
  plt.savefig('L2_'+feature_name+'.png')
  plt.close(fig)
