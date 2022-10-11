import csv
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

feature_groups = ['surface_counts', 'POS_features', 'lexical_sophistication', 'morphological', 'syntactic', 'cohesion']
surface_counts = ['Number of Tokens',
                  'Number of Sentences',
                  'Number of Word Types',
                  'Number of Word Types with More Than 2 Syllables',
                  'Mean Sentence Length in Tokens',
                  'Mean Token Length in Syllables',
                  'Lexical Richness: Type Token Ratio (TTR)',
                  'Lexical Richness: Type Token Ratio (Corrected TTR)',
                  ]
POS_features = ['Lexical Variation Feature: Adjective',
                'Lexical Variation Feature: Adverb',
                'Lexical Variation Feature: Corrected Verb Variation 1',
                'Lexical Variation Feature: Lexical',
                'Lexical Variation Feature: Noun',
                'Lexical Variation Feature: Verb',
                'Number of POS Feature: Adjective Tokens',
                'Number of POS Feature: Adverb Tokens',
                'Number of POS Feature: Lexical word Tokens',
                'Number of POS Feature: Noun Tokens',
                'Number of POS Feature: Punctuation Tokens',
                'POS Density Feature: Article',
                'POS Density Feature: Auxiliary Verb',
                'POS Density Feature: Conjunction',
                'POS Density Feature: Coordinating Conjunction',
                'POS Density Feature: Determiner',
                'POS Density Feature: Functional Words',
                'POS Density Feature: Interjection',
                'POS Density Feature: Lexical Words',
                'POS Density Feature: Modifier',
                'POS Density Feature: Noun',
                'POS Density Feature: Preposition',
                'POS Density Feature: Pronoun',
                'POS Density Feature: Proper Noun',
                'POS Density Feature: Subordinating Conjunction'
                ]
lexical_sophistication = ['Lexical Sophistication Feature: Age of Acquisition (AW Token)',
                          'Lexical Sophistication Feature: Concreteness (AW Token)',
                          'Lexical Sophistication Feature: Familiarity (LW Token)',
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

morphological = ['Number of Morphological Features: Conditional Verb per VP',
                 'Number of Morphological Features: Imperfect Verb per VP',
                 'Number of Morphological Features: Inflected Infinitive Verb per VP',
                 'Number of Morphological Features: Pluperfect Verb per VP',
                 'Number of Morphological Features: Preterite Verb per VP',
                 'Number of Morphological Features: Subjunctive Verb per VP',
                 'Number of Morphological Features: Simple Future Verb per VP',
                 'Number of Morphological Features: Relative Pronouns',
                 'Number of Morphological Features: Indefinite Pronouns',
                 'Morphological Complexity Inflection Feature: Feminine inflection per word token',
                 'Morphological Complexity Inflection Feature: First Person per word token',
                 'Morphological Complexity Inflection Feature: Imperatives per Verb',
                 'Morphological Complexity Inflection Feature: Imperatives per word token',
                 'Morphological Complexity Inflection Feature: Imperfect tense per verb token',
                 'Morphological Complexity Inflection Feature: Indicatives per Verb',
                 'Morphological Complexity Inflection Feature: Indicatives per word token',
                 'Morphological Complexity Inflection Feature: Infinite Verb Density',
                 'Morphological Complexity Inflection Feature: Masculine inflection per word token',
                 'Morphological Complexity Inflection Feature: Past Participle Verb Density',
                 'Morphological Complexity Inflection Feature: Past tense per verb token',
                 'Morphological Complexity Inflection Feature: Second Person per word token',
                 'Morphological Complexity Inflection Feature: Singular per word token',
                 'Morphological Complexity Inflection Feature: Subjunctives per verb token',
                 'Morphological Complexity Inflection Feature: Subjunctives per word token',
                 'Morphological Complexity Inflection Feature: Third Person per word token',
                 ]

syntactic = ['Number of Syntactic Constituents: Passive Sentences',
             'Number of Syntactic Constituents: Coordinate Phrases',
             'Number of Syntactic Constituents: Dependent Clauses',
             'Number of Syntactic Constituents: Relative Clauses',
             'Number of Syntactic Constituents: Subordinate Clauses',
             'Number of Syntactic Constituents: Open Clausal Complement',
             'Number of Syntactic Constituents: Clausal Subject',
             'Syntactic Complexity Feature: Complex Nominals per Sentence',
             'Syntactic Complexity Feature: Complex T-unit Ratio',
             'Syntactic Complexity Feature: Complex T-unit per Sentence',
             'Syntactic Complexity Feature: Coordinate Phrases per Sentence',
             'Syntactic Complexity Feature: Dependent clause ratio',
             'Syntactic Complexity Feature: Dependent clauses per Sentence',
             'Syntactic Complexity Feature: Inverted Pseudoclefts per VP',
             'Syntactic Complexity Feature: It-Clefts per VP',
             'Syntactic Complexity Feature: Mean Length of Clause',
             'Syntactic Complexity Feature: Mean Length of Complex T-unit',
             'Syntactic Complexity Feature: Mean Length of Noun Phrase',
             'Syntactic Complexity Feature: Mean Length of Prepositional Phrase',
             'Syntactic Complexity Feature: Mean Length of T-unit',
             'Syntactic Complexity Feature: Noun Phrases per Sentence',
             'Syntactic Complexity Feature: Prenominal Modifier per Complex Noun Phrase',
             'Syntactic Complexity Feature: Prepositional Phrases per Sentence',
             'Syntactic Complexity Feature: Pseudoclefts per VP',
             'Syntactic Complexity Feature: Relative Clauses per Sentence',
             'Syntactic Complexity Feature: Sentence Complexity Ratio',
             'Syntactic Complexity Feature: Sentence Coordination Ratio',
             'Syntactic Complexity Feature: T-unit complexity ratio',
             'Syntactic Complexity Feature: Verb Cluster per Sentence',
             'Syntactic Complexity Feature: Verb Phrases per Sentence',
             'Syntactic Complexity Feature: e-que Cleft per VP']

cohesion = ['Number of Connectives: Mendes Additive Connectives',
            'Number of Connectives: Mendes All Connectives',
            'Number of Connectives: Mendes Causal Connectives',
            'Number of Connectives: Mendes Concessive Connectives',
            'Number of Connectives: Mendes Single-Word Connectives',
            'Number of Connectives: Mendes Temporal Connectives',
            'Cohesive Complexity Feature: Mendes Additive Connectives per Token',
            'Cohesive Complexity Feature: Mendes All Connectives per Token',
            'Cohesive Complexity Feature: Mendes Causal Connectives per Token',
            'Cohesive Complexity Feature: Mendes Concessive Connectives per Token',
            'Cohesive Complexity Feature: Mendes Single-Word Connectives per Connective']

# L1
df1 = pd.read_csv("../data/School_levels_all_features_noNaN.csv", encoding='unicode_escape')

for feature in cohesion:
    a = df1.loc[df1['Level'] == 1, feature]
    b = df1.loc[df1['Level'] == 2, feature]
    c = df1.loc[df1['Level'] == 3, feature]
    d = df1.loc[df1['Level'] == 3, feature]
    columns = [a, b, c, d]
    # plt.rcParams["figure.figsize"] = (7,7)
    fig, ax = plt.subplots()
    plt.ylabel(feature)
    plt.xlabel('School level')
    ax.boxplot(columns, notch=True, showfliers=True)
    # plt.show()
    feature_name = feature.replace(" ", "_")
    plt.savefig('Readability_' + feature_name + '.png')
    plt.close(fig)


for feature in cohesion:
    a = sns.lineplot(data=df1, x="Level", y=feature)
    a.set_xticks(range(1, 4))
    feature_name = feature.replace(" ", "_")
    # plt.figure()
    plt.savefig('Readability_' + feature_name + 'GT' + '.png')
    plt.close()


# L2
df2 = pd.read_csv("../data/NLI-PT_all_features_new_noNaN.csv", encoding='unicode_escape')

for feature in cohesion:
    a = df2.loc[df2['Proficiency'] == 1, feature]
    b = df2.loc[df2['Proficiency'] == 2, feature]
    c = df2.loc[df2['Proficiency'] == 3, feature]
    columns = [a, b, c]
    # plt.rcParams["figure.figsize"] = (7,7)
    fig, ax = plt.subplots()
    # plt.ylim(0, 1300)
    plt.ylabel(feature)
    plt.xlabel('Proficiency level')
    ax.boxplot(columns, notch=True, showfliers=True)
    #  plt.show()
    feature_name = feature.replace(" ", "_")
    plt.savefig('Proficiency_' + feature_name + '.png')
    plt.close(fig)

for feature in cohesion:
    a = sns.lineplot(data=df2, x="Proficiency", y=feature)
    a.set_xticks(range(1, 4))
    feature_name = feature.replace(" ", "_")
    plt.savefig('Proficiency_' + feature_name + 'GT' + '.png')
    plt.close()
