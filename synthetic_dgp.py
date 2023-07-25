"""
Code for running a semi-synthetic DGP setup to find average causal effect and confidence intervals.
"""

import pandas as pd
import numpy as np
from scipy.special import expit
import statsmodels.api as sm
import scipy.stats as stats
from adjustment import *
from gensim.models import Word2Vec
import pickle

from proximal import *
from regex_predictor import *
from odds_ratio import *
from backdoor import *
from bag_of_words import *

def run_semi_synthetic_dgp(oracle='afib', verbose=False):
    master_data = pd.read_csv('csv_files/master_data.csv')

    # find the candidates for using regular expression matching
    regex_candidates = []

    if oracle == 'afib':
        zero_shot_preds = pd.read_csv('csv_files/predictions-xxl.csv')
        regex_candidates.append('atrial')
        regex_candidates.append('fibrillation')
    elif oracle == 'heart_fail':
        zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-congestiveheartfailure-sentence.csv')
        regex_candidates.append('congestive')
        regex_candidates.append('heart')
        regex_candidates.append('failure')
    elif oracle == 'kidney_fail':
        zero_shot_preds = pd.read_csv('csv_files/predictions-xxl-acutekidneyfailure-sentence.csv')
        regex_candidates.append('acute')
        regex_candidates.append('kidney')
        regex_candidates.append('failure')

    # train a Word2Vec model that can tell you the nearest neighbors for a certain word
    model = Word2Vec.load('word2vec.model')
    similar_words = model.wv.most_similar(positive=regex_candidates)

    for i in range(5):
        regex_candidates.append(similar_words[i][0])

    if verbose:
        print(regex_candidates)

    ace_predictions = []
    conf_intervals = []
    for word in regex_candidates:
        regex_preds = regular_expression_predict(master_data['notes_half2'], [word])

        # if verbose:
        #     print(np.mean(master_data[oracle] == zero_shot_preds['prediction']))
        #     print(np.mean(zero_shot_preds['prediction']))
        #     print(create_confusion_matrix(master_data[oracle], zero_shot_preds['prediction']))
        #     print()

        #     print(np.mean(master_data[oracle] == regex_preds))
        #     print(np.mean(regex_preds))
        #     print(create_confusion_matrix(master_data[oracle], regex_preds))

        semi_synthetic_data = pd.DataFrame({'U': master_data[oracle], 'W': zero_shot_preds['prediction'], 'Z': regex_preds,
                                    'age': master_data['age'], 'gender': master_data['gender']})
        
        # generate semi-synthetic data
        np.random.seed(3)

        size = len(semi_synthetic_data)

        C = np.random.normal(0, 1, size)

        A = np.random.binomial(1, expit(0.8*semi_synthetic_data['U']+C), size)

        Y = np.random.normal(0, 1, size) + 1.3*A + 1.4*semi_synthetic_data['U'] + C

        semi_synthetic_data['A'] = A
        semi_synthetic_data['Y'] = Y
        semi_synthetic_data['C'] = C

        # if verbose:
        #     print(odds_ratio('U', 'W', [], semi_synthetic_data))
        #     print(odds_ratio('U', 'Z', [], semi_synthetic_data))

        #     print()
        #     print(np.mean(semi_synthetic_data['W'] == semi_synthetic_data['Z']))
        #     print()

        #     print(odds_ratio('W', 'Z', ['U'], semi_synthetic_data))
        #     print(odds_ratio('W', 'Z', ['U', 'age', 'gender'], semi_synthetic_data))

        # approximate the ACE
        ace_predictions.append(proximal_find_ace('A', 'Y', 'W', 'Z', ['C'], semi_synthetic_data))
        conf_intervals.append(compute_confidence_intervals("A", "Y", "W", "Z", ['C'], semi_synthetic_data))

    return (ace_predictions, conf_intervals)