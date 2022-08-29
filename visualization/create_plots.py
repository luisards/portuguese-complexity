#OPEN FILES

#L1
for i in range(36):
  feature = df.columns[i+2]
  df.groupby(['Level'])[feature].mean().plot()
  plt.title(feature+' per Level')
  plt.ylabel(feature)
  plt.xlabel('School level')
  plt.show()
  print(" ")

#L2
lexical_sophistication = ['Lexical Sophistication Feature: Age of Acquisition (AW Token)',
'Lexical Sophistication Feature: Concreteness (AW Token)',
'Lexical Sophistication Feature: Familiarity (AW Token)',
'Lexical Sophistication Feature: Imageability (AW Token)',
'Lexical Sophistication Feature: SUBTLEX Contextual Diversity (AW Token)',
'Lexical Sophistication Feature: SUBTLEX Frequency Band 1',
'Lexical Sophistication Feature: SUBTLEX Frequency Band 2',
'Lexical Sophistication Feature: SUBTLEX Frequency Band 3',
'Lexical Sophistication Feature: SUBTLEX Frequency Band 4',
'Lexical Sophistication Feature: SUBTLEX Frequency Band 5',
'Lexical Sophistication Feature: SUBTLEX Frequency Band 6',
'Lexical Sophistication Feature: SUBTLEX Frequency Band 7',
'Lexical Sophistication Feature: SUBTLEX Frequency Top 1000',
'Lexical Sophistication Feature: SUBTLEX Frequency Top 2000',
'Lexical Sophistication Feature: SUBTLEX Frequency Top 3000',
'Lexical Sophistication Feature: SUBTLEX Frequency Top 4000',
'Lexical Sophistication Feature: SUBTLEX Frequency Top 5000',
'Lexical Sophistication Feature: SUBTLEX Frequency Top 6000 and Below',
'Lexical Sophistication Feature: SUBTLEX Logarithmic Contextual Diversity (AW Token)',
'Lexical Sophistication Feature: SUBTLEX Logarithmic Word Frequency (AW Token)',
'Lexical Sophistication Feature: SUBTLEX Word Frequency per Million (AW Token)'
]

for feature in lexical_sophistication:
  a = df.loc[df['Proficiency'] == 1, feature]
  b = df.loc[df['Proficiency'] == 2, feature]
  c = df.loc[df['Proficiency'] == 3, feature]
  columns = [a, b, c]
  #plt.rcParams["figure.figsize"] = (7,7)
  fig, ax = plt.subplots()
  #plt.ylim(0, 1300)
  plt.ylabel(feature)
  plt.xlabel('Proficiency level')
  ax.boxplot(columns, notch=True, showfliers=True)
  plt.show()
